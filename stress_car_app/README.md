# Stress-Aware Driving System — Prototype

A three-layer prototype: a BLE heart rate sensor feeds live data through
an AI decision layer, which a web dashboard visualizes in real time.

```
BLE Sensor → Guardrails → AI Layer → Agent → Backend (WebSocket) → Frontend
                              ↓
                        Observability
```

## Layers

- **`backend/ai_layer.py`** — the AI layer. Pure decision logic: takes a
  `SensorReading` in, returns a `StressAssessment` out (stress score,
  target speed, whether the AI is intervening). No networking or BLE code
  lives here on purpose — this is the file you'll extend later (add a
  camera-based score, swap the formula, plug in a trained model).

- **`backend/ble_sensor.py`** — sensor input adapter. The only file that
  knows about Bluetooth. Swapping to a different sensor or protocol later
  means changing this file only.

- **`backend/server.py`** — the backend. Runs the BLE adapter in the
  background, feeds readings into the AI layer, and broadcasts results to
  connected browsers over a WebSocket at `/ws`. Also serves the frontend.
  Falls back to **demo mode** (synthetic wandering heart rate) if no real
  sensor is found within ~10 seconds, so you can see the whole pipeline
  working before you have hardware.

- **`frontend/index.html`** — the dashboard. Connects to the backend over
  WebSocket and renders live speed, a stress gauge, and AI intervention
  status. Includes a manual override slider for testing without a sensor,
  plus an agent status strip and live metrics.

- **`backend/guardrails.py`** — safety layer around the AI. Rejects
  physiologically implausible sensor readings (e.g. BPM 300), detects
  stale data and falls back to a safe non-intervening state, and rate-
  limits how fast the AI's requested speed can change per tick so a
  single noisy reading can't cause an abrupt jolt. Deliberately simple,
  deterministic code — guardrails should be the part of the system that's
  easiest to trust and fastest to reason about.

- **`backend/observability.py`** — structured logging for every decision
  the pipeline makes. Writes an append-only JSONL event log
  (`backend/events.jsonl`) and keeps rolling metrics (intervention rate,
  rejected readings, average stress score) exposed at `GET /metrics`.
  Recent raw events are available at `GET /events`.

- **`backend/agent.py`** — the agentic layer. Instead of reacting to a
  single instantaneous stress score, it tracks *sustained* elevation over
  time and moves through states: `normal → watch → intervene → escalate`.
  Only sustained, high stress triggers escalation actions (alert driver,
  suggest pulling over, log an incident) — a brief spike just gets
  watched, not acted on.

- **`backend/evals.py`** — a standalone eval harness. Run
  `python evals.py` any time you change the scoring formula, thresholds,
  or agent logic, to check known scenarios still behave as expected
  (e.g. "a calm profile never intervenes", "sustained extreme stress
  eventually escalates", "an implausible reading gets rejected").

- **`backend/emergency.py`** — notifies a designated **personal emergency
  contact** (never 911) when the agent's `ESCALATE` state goes
  unacknowledged. On escalation, the dashboard shows a 15-second
  confirmation countdown ("I'm OK — Cancel"); if the driver doesn't
  respond, the contact is texted automatically (same logic as Apple
  Watch fall detection — no response may mean they can't respond). Runs
  in a safe **simulated mode** by default (just logs what would have
  been sent) unless you configure real Twilio credentials — see below.
  Calling 911 automatically was deliberately left out: a heuristic
  stress score isn't a validated emergency trigger, and false automated
  emergency calls carry real legal and safety consequences. The
  dashboard has an always-visible **manual** "Call 911" button instead,
  for the driver or a bystander to use themselves.

### Enabling real emergency notifications (optional)

By default, `emergency.py` runs in simulated mode — safe to test, no
external services required, just logs what would have been sent.

**Email (free, easiest to set up first):**

```bash
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="you@gmail.com"
export SMTP_PASSWORD="..."             # Gmail: use an App Password, not your real password
export SMTP_FROM_EMAIL="you@gmail.com"
export EMERGENCY_CONTACT_NAME="..."
export EMERGENCY_CONTACT_EMAIL="..."   # who receives the alert
```

**SMS via [Twilio](https://www.twilio.com/) (small per-message cost):**

```bash
export TWILIO_ACCOUNT_SID="..."
export TWILIO_AUTH_TOKEN="..."
export TWILIO_FROM_NUMBER="+1..."      # a number you own on Twilio
export EMERGENCY_CONTACT_NAME="..."
export EMERGENCY_CONTACT_PHONE="+1..." # who gets texted
pip install twilio
```

Either channel can be enabled independently, or both together — if both
are configured, an escalation notifies via both email and SMS. Without
any of this set, the app still runs the full escalation → countdown →
notification flow — it just logs the simulated message instead of
sending anything real. Check `GET /emergency-status` any time to see
which channels are currently active.

## Running it

```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload
```

Then open **http://localhost:8000** in your browser.

- If a BLE heart rate sensor (Polar H10, Wahoo Tickr, etc.) is on and
  worn, the server will find and connect to it automatically.
- If not, it switches to demo mode after ~10 seconds so the dashboard
  still has live (synthetic) data to show.
- Use the **Manual override** slider at the bottom of the dashboard any
  time to test specific stress levels directly, regardless of sensor state.
- Watch the **agent strip** at the bottom to see the escalation state
  change as you hold the override slider above 60 for a few seconds.
- Check `http://localhost:8000/metrics` any time for a live JSON snapshot
  of pipeline health.

## Running the evals

```bash
cd backend
python evals.py
```

This runs 6 scenarios against the real `StressEngine` + `SafetyGuardrails`
+ `DriverSafetyAgent` code (not mocks) and prints a pass/fail report,
e.g.:

```
[PASS] calm_driver_no_intervention
[PASS] sustained_stress_triggers_intervention
[PASS] extreme_sustained_stress_escalates
[PASS] agent_resets_when_stress_drops
[PASS] guardrail_rejects_implausible_reading
[PASS] speed_rate_limiter_smooths_jumps
6/6 scenarios passed
```

## Extending it

- **Add the camera-based stress score**: create a `SensorReading`-like
  input in a new adapter file (e.g. `camera_sensor.py`), and combine it
  with the BLE score inside `ai_layer.py` — e.g. a weighted average in
  `StressEngine.ingest()`. Nothing in `server.py` or the frontend needs
  to change for the score calculation itself, only how you feed in and
  combine sources.
- **Replace the scoring formula**: everything's in
  `StressEngine._compute_stress_score()` — swap it for a trained model
  or a different heuristic without touching the rest of the app.
- **Persist history**: the tick loop in `server.py` is a natural place to
  also write each assessment to a file or database for later review.
