"""
AI Layer
--------
Pure decision-making logic: takes raw sensor readings in, produces a
stress score and a speed decision out. Deliberately has zero networking,
BLE, or UI code in it -- that separation is the whole point. This is the
module you'll extend over time:

  - swap in a smarter stress-scoring formula
  - add a second input source (e.g. camera-based facial stress score)
  - replace the threshold logic with a trained classifier

Nothing outside this file needs to change when you do.
"""

import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

STRESS_MAX = 100.0
BASELINE_WINDOW_SECONDS = 15
RR_WINDOW_SIZE = 20

MAX_SPEED = 6.0
MIN_SPEED = 1.0
STRESS_THRESHOLD = 60.0

# Manual-override / demo mode: when the dashboard slider sets a stress score
# directly, the heart-rate vitals shown must move with it. These constants map
# a 0-100 stress score to a simulated BPM and RMSSD so the display is consistent.
SIM_RESTING_BPM = 70.0          # used if no baseline has been calibrated yet
SIM_MAX_ELEVATION_BPM = 60.0    # score 100 -> resting + 60 bpm (about 130 bpm)
SIM_RMSSD_CALM_MS = 50.0        # HRV at score 0
SIM_RMSSD_STRESSED_MS = 10.0    # HRV at score 100 (low HRV = high stress)


@dataclass
class SensorReading:
    """One incoming reading from a sensor (currently: BLE heart rate)."""
    bpm: int
    rr_intervals_ms: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class StressAssessment:
    """Output of the AI layer for one point in time."""
    stress_score: float          # 0-100
    bpm: Optional[int]
    baseline_bpm: Optional[float]
    rmssd: Optional[float]
    target_speed: float          # what the AI wants the car's speed to be
    intervening: bool            # whether AI is actively slowing the car
    simulated: bool = False      # True when BPM/RMSSD are simulated from a manual score


class StressEngine:
    """
    Stateful engine: accumulates readings over time (needed for baseline
    calibration and HRV calculation), and produces a StressAssessment on
    demand.

    NOTE ON THE SCORING FORMULA:
    This is a simple, transparent heuristic -- not a validated clinical
    stress metric. It combines:
      1) BPM elevation above a self-calibrated resting baseline
      2) RMSSD of RR intervals (heart rate variability) -- lower
         variability tends to correlate with higher sympathetic/stress
         activation
    Treat this as a reasonable starting point to tune once you have real
    data, or to replace outright with a trained model later.
    """

    def __init__(self, threshold: float = STRESS_THRESHOLD, clock: Optional[callable] = None):
        """
        clock: callable returning the current time (defaults to time.time).
        Injectable so eval/test code can fast-forward the baseline
        calibration window deterministically instead of sleeping in real
        time -- see evals.py.
        """
        self.threshold = threshold
        self._clock = clock or time.time
        self._start_time = self._clock()
        self._baseline_samples: List[int] = []
        self.baseline_bpm: Optional[float] = None
        self._rr_window: List[float] = []
        self.last_bpm: Optional[int] = None

    def ingest(self, reading: SensorReading) -> StressAssessment:
        self.last_bpm = reading.bpm

        # --- baseline calibration (first N seconds of data) ---
        if self.baseline_bpm is None:
            elapsed = self._clock() - self._start_time
            if elapsed <= BASELINE_WINDOW_SECONDS:
                self._baseline_samples.append(reading.bpm)
            else:
                self.baseline_bpm = (
                    statistics.mean(self._baseline_samples)
                    if self._baseline_samples
                    else reading.bpm
                )

        # --- rolling RR window for HRV ---
        if reading.rr_intervals_ms:
            self._rr_window.extend(reading.rr_intervals_ms)
            self._rr_window = self._rr_window[-RR_WINDOW_SIZE:]

        score, rmssd = self._compute_stress_score()
        target_speed, intervening = self._decide_speed(score)

        return StressAssessment(
            stress_score=score,
            bpm=reading.bpm,
            baseline_bpm=self.baseline_bpm,
            rmssd=rmssd,
            target_speed=target_speed,
            intervening=intervening,
        )

    def _compute_stress_score(self):
        score = 0.0
        rmssd = None

        # BPM elevation component (0-60 points)
        if self.baseline_bpm and self.last_bpm:
            elevation = self.last_bpm - self.baseline_bpm
            bpm_component = max(0.0, min(60.0, (elevation / 30.0) * 60.0))
            score += bpm_component

        # HRV / RMSSD component (0-40 points)
        if len(self._rr_window) >= 4:
            diffs = [
                self._rr_window[i + 1] - self._rr_window[i]
                for i in range(len(self._rr_window) - 1)
            ]
            rmssd = (statistics.mean([d ** 2 for d in diffs])) ** 0.5

            if rmssd <= 15:
                hrv_component = 40.0
            elif rmssd >= 60:
                hrv_component = 0.0
            else:
                hrv_component = 40.0 * (60.0 - rmssd) / (60.0 - 15.0)

            score += hrv_component

        return max(0.0, min(STRESS_MAX, score)), rmssd

    def _decide_speed(self, stress_score: float):
        if stress_score >= self.threshold:
            overflow = (stress_score - self.threshold) / (STRESS_MAX - self.threshold)
            overflow = min(1.0, overflow)
            target = MAX_SPEED - overflow * (MAX_SPEED - MIN_SPEED)
            return target, True
        return MAX_SPEED, False

    def simulated_vitals_for_score(self, score: float):
        """
        Map a 0-100 stress score to a plausible (bpm, rmssd_ms) pair, so that a
        high manual stress score is shown with a significantly elevated heart
        rate and suppressed HRV instead of the last calm demo reading.
        """
        fraction = max(0.0, min(1.0, score / STRESS_MAX))
        resting = self.baseline_bpm or SIM_RESTING_BPM
        bpm = int(round(resting + fraction * SIM_MAX_ELEVATION_BPM))
        rmssd = SIM_RMSSD_CALM_MS - fraction * (SIM_RMSSD_CALM_MS - SIM_RMSSD_STRESSED_MS)
        return bpm, rmssd

    def manual_assessment(self, manual_score: float) -> StressAssessment:
        """
        Used when overriding with a manual/test score instead of live sensor data.
        BPM and RMSSD are simulated from the score so the dashboard stays
        consistent (high stress -> high BPM, low HRV). Flagged simulated=True.
        """
        target_speed, intervening = self._decide_speed(manual_score)
        bpm, rmssd = self.simulated_vitals_for_score(manual_score)
        self.last_bpm = bpm
        return StressAssessment(
            stress_score=manual_score,
            bpm=bpm,
            baseline_bpm=self.baseline_bpm,
            rmssd=rmssd,
            target_speed=target_speed,
            intervening=intervening,
            simulated=True,
        )
