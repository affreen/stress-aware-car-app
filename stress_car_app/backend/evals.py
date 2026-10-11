"""
Eval Harness
------------
A standalone test suite for the AI pipeline (StressEngine + guardrails +
agent). Run this whenever you change the scoring formula, thresholds, or
agent logic, to catch regressions before they reach the dashboard.

Not a unit-test framework dependency on purpose -- kept dependency-free
so it's easy to run anywhere:

    python evals.py

Each scenario feeds a sequence of synthetic SensorReadings through the
real pipeline and asserts properties of the output (e.g. "a calm,
resting profile should never trigger intervention", "a sustained high
stress profile should eventually escalate").

For scenarios that depend on elapsed time (the agent's watch/escalate
timers), we directly adjust the agent's internal clock reference instead
of sleeping in real time -- see `_fast_forward()`. This keeps the eval
suite fast (seconds, not minutes) while still exercising the real logic.
"""

import time
from dataclasses import dataclass
from typing import Callable, List

from ai_layer import StressEngine, SensorReading, BASELINE_WINDOW_SECONDS
from guardrails import SafetyGuardrails
from agent import DriverSafetyAgent, AgentState


@dataclass
class EvalResult:
    name: str
    passed: bool
    detail: str = ""


class FakeClock:
    """A controllable clock so evals can fast-forward time deterministically
    instead of sleeping in real time (used for StressEngine baseline calibration)."""
    def __init__(self, start: float = 1_000_000.0):
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float):
        self.t += seconds


def _fast_forward(agent: DriverSafetyAgent, seconds: float):
    """Simulate elapsed time for the agent's escalation timer without sleeping."""
    if agent._elevated_since is not None:
        agent._elevated_since -= seconds


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------
def calm_readings(n=20, bpm=70):
    return [SensorReading(bpm=bpm, rr_intervals_ms=[857.0, 840.0]) for _ in range(n)]


def elevated_readings(n=20, bpm=125):
    # High BPM, low RR variability (low HRV) -- consistent with a stress profile
    return [SensorReading(bpm=bpm, rr_intervals_ms=[480.0, 478.0]) for _ in range(n)]


def implausible_reading():
    return SensorReading(bpm=300, rr_intervals_ms=[100.0])


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def scenario_calm_driver_no_intervention() -> EvalResult:
    clock = FakeClock()
    engine = StressEngine(clock=clock.now)
    guardrails = SafetyGuardrails()
    last_assessment = None

    for reading in calm_readings():
        check = guardrails.check_reading(reading)
        if not check.ok:
            return EvalResult("calm_driver_no_intervention", False, f"unexpected rejection: {check.reason}")
        clock.advance(1.0)
        last_assessment = engine.ingest(reading)

    passed = last_assessment is not None and not last_assessment.intervening and last_assessment.stress_score < 40
    return EvalResult(
        "calm_driver_no_intervention",
        passed,
        f"final score={last_assessment.stress_score:.1f}, intervening={last_assessment.intervening}",
    )


def scenario_sustained_stress_triggers_intervention() -> EvalResult:
    clock = FakeClock()
    engine = StressEngine(clock=clock.now)
    agent = DriverSafetyAgent(threshold=60.0)

    # Establish baseline with calm data first (advance clock past the
    # calibration window so baseline_bpm actually finalizes)
    for reading in calm_readings(n=16):
        clock.advance(1.0)
        engine.ingest(reading)

    last_decision = None
    for reading in elevated_readings(n=5):
        clock.advance(1.0)
        assessment = engine.ingest(reading)
        decision = agent.step(assessment.stress_score, assessment.target_speed)
        _fast_forward(agent, 2.0)  # simulate 2 seconds passing each tick
        last_decision = decision

    passed = last_decision is not None and last_decision.state in (AgentState.INTERVENE, AgentState.ESCALATE)
    return EvalResult(
        "sustained_stress_triggers_intervention",
        passed,
        f"final agent state={last_decision.state if last_decision else None}",
    )


def scenario_extreme_sustained_stress_escalates() -> EvalResult:
    clock = FakeClock()
    engine = StressEngine(clock=clock.now)
    agent = DriverSafetyAgent(threshold=60.0)

    for reading in calm_readings(n=16):
        clock.advance(1.0)
        engine.ingest(reading)

    last_decision = None
    all_actions: List[str] = []
    for reading in elevated_readings(n=15, bpm=140):
        clock.advance(1.0)
        assessment = engine.ingest(reading)
        decision = agent.step(assessment.stress_score, assessment.target_speed)
        all_actions.extend(decision.actions)
        _fast_forward(agent, 3.0)  # simulate 3 seconds passing each tick
        last_decision = decision

    passed = last_decision is not None and last_decision.state == AgentState.ESCALATE
    passed = passed and "alert_driver" in all_actions and "suggest_pull_over" in all_actions
    return EvalResult(
        "extreme_sustained_stress_escalates",
        passed,
        f"final state={last_decision.state if last_decision else None}, actions_fired={set(all_actions)}",
    )


def scenario_agent_resets_when_stress_drops() -> EvalResult:
    clock = FakeClock()
    engine = StressEngine(clock=clock.now)
    agent = DriverSafetyAgent(threshold=60.0)

    for reading in calm_readings(n=16):
        clock.advance(1.0)
        engine.ingest(reading)

    for reading in elevated_readings(n=5):
        clock.advance(1.0)
        assessment = engine.ingest(reading)
        agent.step(assessment.stress_score, assessment.target_speed)
        _fast_forward(agent, 2.0)

    # Now stress drops back down
    final_decision = None
    for reading in calm_readings(n=5):
        clock.advance(1.0)
        assessment = engine.ingest(reading)
        final_decision = agent.step(assessment.stress_score, assessment.target_speed)

    passed = final_decision is not None and final_decision.state == AgentState.NORMAL
    return EvalResult(
        "agent_resets_when_stress_drops",
        passed,
        f"final state={final_decision.state if final_decision else None}",
    )


def scenario_guardrail_rejects_implausible_reading() -> EvalResult:
    guardrails = SafetyGuardrails()
    check = guardrails.check_reading(implausible_reading())
    passed = not check.ok
    return EvalResult("guardrail_rejects_implausible_reading", passed, f"reason={check.reason}")


def scenario_speed_rate_limiter_smooths_jumps() -> EvalResult:
    guardrails = SafetyGuardrails()
    # Request an abrupt drop from 6.0 to 1.0 in one step
    first = guardrails.apply_speed_limit(1.0)
    passed = first > 1.0  # should not jump straight to the minimum in one tick
    return EvalResult(
        "speed_rate_limiter_smooths_jumps",
        passed,
        f"first-tick output={first:.2f} (should be > 1.0, i.e. not an instant jump)",
    )


SCENARIOS: List[Callable[[], EvalResult]] = [
    scenario_calm_driver_no_intervention,
    scenario_sustained_stress_triggers_intervention,
    scenario_extreme_sustained_stress_escalates,
    scenario_agent_resets_when_stress_drops,
    scenario_guardrail_rejects_implausible_reading,
    scenario_speed_rate_limiter_smooths_jumps,
]


def run_all():
    results = [scenario() for scenario in SCENARIOS]

    print("=" * 70)
    print("AI PIPELINE EVAL REPORT")
    print("=" * 70)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}")
        if r.detail:
            print(f"       {r.detail}")

    passed_count = sum(1 for r in results if r.passed)
    print("-" * 70)
    print(f"{passed_count}/{len(results)} scenarios passed")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_all()
