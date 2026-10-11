"""
Guardrails
----------
Safety checks that sit between raw sensor data / AI output and anything
that acts on it. None of this is "AI" -- it's deliberately dumb, simple,
deterministic code, because guardrails should be the part of the system
you trust the most and the fastest to reason about.

Three jobs:
  1. Reject physiologically implausible sensor readings (sensor glitches,
     disconnection artifacts) before they reach the AI layer.
  2. Detect stale data -- if no reading has arrived recently, don't keep
     acting on old data silently.
  3. Rate-limit how fast the AI is allowed to change the car's target
     speed, so a single bad/noisy reading can't cause an abrupt jolt.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from ai_layer import SensorReading, MAX_SPEED, MIN_SPEED

# Plausible human heart rate bounds while conscious and driving.
MIN_PLAUSIBLE_BPM = 30
MAX_PLAUSIBLE_BPM = 220

# Plausible RR interval bounds in ms (roughly 30-220 bpm equivalent)
MIN_RR_MS = 270
MAX_RR_MS = 2000

STALE_DATA_SECONDS = 8.0          # no reading for this long -> considered stale
MAX_SPEED_DELTA_PER_TICK = 0.15   # max change in target speed per tick (safety limiter)


@dataclass
class GuardrailResult:
    ok: bool
    reason: Optional[str] = None


class SensorGuardrail:
    """Validates incoming sensor readings before they reach the AI layer."""

    def __init__(self):
        self.last_valid_reading_time: Optional[float] = None
        self.rejected_count = 0

    def validate(self, reading: SensorReading) -> GuardrailResult:
        if not (MIN_PLAUSIBLE_BPM <= reading.bpm <= MAX_PLAUSIBLE_BPM):
            self.rejected_count += 1
            return GuardrailResult(False, f"BPM {reading.bpm} outside plausible range")

        for rr in reading.rr_intervals_ms:
            if not (MIN_RR_MS <= rr <= MAX_RR_MS):
                self.rejected_count += 1
                return GuardrailResult(False, f"RR interval {rr:.0f}ms outside plausible range")

        self.last_valid_reading_time = reading.timestamp
        return GuardrailResult(True)

    def is_stale(self) -> bool:
        if self.last_valid_reading_time is None:
            return False  # no data yet is a different state than stale data
        return (time.time() - self.last_valid_reading_time) > STALE_DATA_SECONDS


class SpeedRateLimiter:
    """
    Prevents the AI's target speed output from jumping abruptly.
    A noisy single reading (or a bug) can't cause a sudden speed change --
    it has to be sustained across multiple ticks to fully take effect.
    """

    def __init__(self, max_delta: float = MAX_SPEED_DELTA_PER_TICK):
        self.max_delta = max_delta
        self.current_output: float = MAX_SPEED

    def limit(self, requested_speed: float) -> float:
        requested_speed = max(MIN_SPEED, min(MAX_SPEED, requested_speed))
        delta = requested_speed - self.current_output
        if delta > self.max_delta:
            delta = self.max_delta
        elif delta < -self.max_delta:
            delta = -self.max_delta
        self.current_output += delta
        return self.current_output


class SafetyGuardrails:
    """
    Bundles the sensor validator and speed rate limiter, and defines the
    fallback behavior when data is stale or missing: default to a safe,
    non-intervening state rather than freezing on old data or guessing.
    """

    def __init__(self):
        self.sensor_guardrail = SensorGuardrail()
        self.speed_limiter = SpeedRateLimiter()

    def check_reading(self, reading: SensorReading) -> GuardrailResult:
        return self.sensor_guardrail.validate(reading)

    def apply_speed_limit(self, requested_speed: float) -> float:
        return self.speed_limiter.limit(requested_speed)

    def fallback_state(self) -> Tuple[float, str]:
        """Called when data is stale/missing. Returns (safe_speed, reason)."""
        safe_speed = self.apply_speed_limit(MAX_SPEED)
        return safe_speed, "No recent valid sensor data — defaulting to normal speed, no intervention"

    def status(self) -> dict:
        return {
            "stale": self.sensor_guardrail.is_stale(),
            "rejected_readings": self.sensor_guardrail.rejected_count,
        }
