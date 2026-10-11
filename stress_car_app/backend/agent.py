"""
Agentic Layer
-------------
Everything so far reacts to a single instantaneous stress score. This
layer adds an agent that reasons over TIME: it tracks how long stress has
been elevated, moves through escalating states, and takes different
actions ("tools") at each stage rather than one flat threshold response.

States:
  NORMAL     - stress below threshold
  WATCH      - stress elevated but brief; agent watches, takes no action yet
  INTERVENE  - stress sustained past a watch period; AI slows the car
               (this is the behavior the rest of the app already had)
  ESCALATE   - stress sustained even longer / very high; agent takes
               additional actions beyond slowing down (alert, suggest
               pull-over, log an incident for later review)

The "tools" the agent can call are just plain methods here (alert,
log_incident, suggest_pull_over) -- in a more complete system these might
dispatch to a real notification service, a car's HMI, or an emergency
contact integration. Keeping them as simple, inspectable functions makes
the agent's behavior easy to test and reason about.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

WATCH_TO_INTERVENE_SECONDS = 5.0     # sustained elevated stress before AI slows car
INTERVENE_TO_ESCALATE_SECONDS = 20.0  # sustained high stress before escalation
ESCALATE_STRESS_MIN = 80.0            # only escalate if stress is quite high, not just sustained


class AgentState(str, Enum):
    NORMAL = "normal"
    WATCH = "watch"
    INTERVENE = "intervene"
    ESCALATE = "escalate"


@dataclass
class AgentDecision:
    state: AgentState
    actions: List[str] = field(default_factory=list)
    target_speed_override: Optional[float] = None
    reasoning: str = ""


class DriverSafetyAgent:
    """
    Call `step(stress_score, base_target_speed)` once per tick. It tracks
    elevation duration internally and returns an AgentDecision describing
    the state, any actions taken, and (optionally) a speed override.
    """

    def __init__(self, threshold: float = 60.0):
        self.threshold = threshold
        self.state = AgentState.NORMAL
        self._elevated_since: Optional[float] = None
        self._last_actions_taken: set = set()  # avoid re-firing the same one-shot action every tick

    def step(self, stress_score: float, base_target_speed: float) -> AgentDecision:
        now = time.time()

        if stress_score < self.threshold:
            # Reset -- stress has come back down
            self._elevated_since = None
            self._last_actions_taken.clear()
            self.state = AgentState.NORMAL
            return AgentDecision(
                state=self.state,
                actions=[],
                target_speed_override=None,
                reasoning="Stress below threshold; no action needed.",
            )

        if self._elevated_since is None:
            self._elevated_since = now

        elevated_duration = now - self._elevated_since
        actions: List[str] = []

        if elevated_duration < WATCH_TO_INTERVENE_SECONDS:
            self.state = AgentState.WATCH
            reasoning = (
                f"Stress score {stress_score:.0f} above threshold for "
                f"{elevated_duration:.1f}s -- watching before acting."
            )
            return AgentDecision(state=self.state, actions=actions, reasoning=reasoning)

        # Sustained past the watch period -> intervene (slow the car)
        self.state = AgentState.INTERVENE
        actions.append("slow_down")
        reasoning = (
            f"Stress sustained above threshold for {elevated_duration:.1f}s -- "
            f"reducing target speed."
        )

        # Further escalation: sustained even longer AND quite high
        if elevated_duration >= INTERVENE_TO_ESCALATE_SECONDS and stress_score >= ESCALATE_STRESS_MIN:
            self.state = AgentState.ESCALATE

            if "alert_driver" not in self._last_actions_taken:
                actions.append("alert_driver")
                self._last_actions_taken.add("alert_driver")

            if "suggest_pull_over" not in self._last_actions_taken:
                actions.append("suggest_pull_over")
                self._last_actions_taken.add("suggest_pull_over")

            if "log_incident" not in self._last_actions_taken:
                actions.append("log_incident")
                self._last_actions_taken.add("log_incident")

            reasoning = (
                f"Stress score {stress_score:.0f} sustained for {elevated_duration:.1f}s -- "
                f"escalating: alerting driver and logging incident."
            )

        return AgentDecision(state=self.state, actions=actions, target_speed_override=None, reasoning=reasoning)

    # --- "tools" the agent can invoke -----------------------------------
    # Kept as simple, side-effect-free-by-default stubs so behavior is
    # predictable and testable. Wire these up to real integrations later
    # (push notification, car HMI, SMS to an emergency contact, etc).

    @staticmethod
    def alert_driver() -> str:
        return "Audible + visual alert shown to driver"

    @staticmethod
    def suggest_pull_over() -> str:
        return "Dashboard message: 'Consider pulling over when safe'"

    @staticmethod
    def log_incident() -> str:
        return "Incident flagged for later review"
