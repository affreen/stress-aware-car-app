"""
Observability
-------------
Structured logging + metrics for the AI pipeline. The goal: every decision
the system makes should be traceable after the fact -- what data came in,
what was decided, and why -- not just visible as a live number on a
dashboard that's gone the next second.

Two things:
  1. An append-only JSONL event log (one JSON object per line) written to
     disk, so you can replay/analyze sessions later.
  2. An in-memory metrics summary for a live `/metrics` endpoint.
"""

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Optional

LOG_FILE = Path(__file__).parent / "events.jsonl"
RING_BUFFER_SIZE = 500  # keep last N events in memory for quick access


@dataclass
class Event:
    timestamp: float
    event_type: str          # "assessment", "guardrail_reject", "agent_action", etc.
    data: dict


class ObservabilityLogger:
    def __init__(self, log_file: Path = LOG_FILE):
        self.log_file = log_file
        self.buffer: Deque[Event] = deque(maxlen=RING_BUFFER_SIZE)

        # Running counters for /metrics
        self.total_readings = 0
        self.total_rejected = 0
        self.total_interventions = 0
        self.total_agent_escalations = 0
        self.stress_scores: Deque[float] = deque(maxlen=200)  # for rolling avg
        self.start_time = time.time()

        # Open in append mode; create if missing
        try:
            self.log_file.touch(exist_ok=True)
        except Exception:
            pass  # if filesystem is read-only in some deployment, don't crash

    def _write(self, event: Event):
        self.buffer.append(event)
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(asdict(event)) + "\n")
        except Exception:
            pass  # observability logging should never crash the app

    def log_assessment(self, assessment_dict: dict):
        self.total_readings += 1
        if assessment_dict.get("intervening"):
            self.total_interventions += 1
        score = assessment_dict.get("stress_score")
        if score is not None:
            self.stress_scores.append(score)
        self._write(Event(time.time(), "assessment", assessment_dict))

    def log_guardrail_reject(self, reason: str, raw_reading: dict):
        self.total_rejected += 1
        self._write(Event(time.time(), "guardrail_reject", {"reason": reason, "reading": raw_reading}))

    def log_agent_action(self, agent_state: str, actions: list, context: dict):
        if agent_state in ("escalate",):
            self.total_agent_escalations += 1
        self._write(Event(time.time(), "agent_action", {
            "agent_state": agent_state,
            "actions": actions,
            "context": context,
        }))

    def get_metrics(self) -> dict:
        uptime = time.time() - self.start_time
        avg_stress = sum(self.stress_scores) / len(self.stress_scores) if self.stress_scores else None
        return {
            "uptime_seconds": round(uptime, 1),
            "total_readings": self.total_readings,
            "total_rejected_readings": self.total_rejected,
            "total_interventions": self.total_interventions,
            "total_agent_escalations": self.total_agent_escalations,
            "intervention_rate": round(self.total_interventions / self.total_readings, 3) if self.total_readings else 0,
            "rolling_avg_stress_score": round(avg_stress, 1) if avg_stress is not None else None,
        }

    def get_recent_events(self, n: int = 20, event_type: Optional[str] = None) -> list:
        events = list(self.buffer)
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return [asdict(e) for e in events[-n:]]
