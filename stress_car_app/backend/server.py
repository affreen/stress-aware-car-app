"""
Backend Server
--------------
FastAPI app that:
  1. Runs the BLE sensor adapter in the background
  2. Feeds each reading into the AI layer
  3. Broadcasts the resulting stress assessment to connected browser
     clients over a WebSocket
  4. Serves the frontend dashboard

Run with:
    pip install fastapi uvicorn bleak websockets
    uvicorn server:app --reload

Then open http://localhost:8000 in your browser.

If no BLE sensor is found within ~10s, the server falls back to DEMO
MODE: it generates a slowly wandering fake heart rate so you can see
the whole pipeline working end-to-end before you have hardware, and so
the frontend has something to render. Manual override from the frontend
(a slider) always takes priority over both.
"""

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ai_layer import StressEngine, SensorReading
from ble_sensor import BleHeartRateSensor
from guardrails import SafetyGuardrails
from observability import ObservabilityLogger
from agent import DriverSafetyAgent, AgentState
from emergency import EmergencyNotifier

app = FastAPI()

engine = StressEngine()
guardrails = SafetyGuardrails()
observability = ObservabilityLogger()
safety_agent = DriverSafetyAgent()
emergency_notifier = EmergencyNotifier()
connected_clients: Set[WebSocket] = set()

EMERGENCY_COUNTDOWN_SECONDS = 15.0

# Shared runtime state
state_lock = asyncio.Lock()
latest_status_text = "Starting up..."
manual_override = {"active": False, "value": 0.0}
demo_mode = {"active": False, "bpm": 72.0}
emergency_prompt = {"pending": False, "acknowledged": False, "prompted_at": None, "reason": ""}


# ---------------------------------------------------------------------------
# Frontend static files
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


@app.get("/metrics")
async def metrics():
    """AI observability endpoint: rolling metrics on the pipeline's behavior."""
    return {
        **observability.get_metrics(),
        "guardrail_status": guardrails.status(),
    }


@app.get("/events")
async def events(n: int = 20, type: str = None):
    """Recent structured events (assessments, guardrail rejects, agent actions)."""
    return observability.get_recent_events(n=n, event_type=type)


@app.get("/emergency-status")
async def emergency_status():
    """Debug endpoint: current emergency contact config and pending prompt state."""
    return {
        "contact_configured": emergency_notifier.contact is not None,
        "real_sms_configured": emergency_notifier.is_configured_for_real_sms(),
        "real_email_configured": emergency_notifier.is_configured_for_real_email(),
        "prompt_pending": emergency_prompt["pending"],
    }


# ---------------------------------------------------------------------------
# WebSocket: pushes live assessments to the browser
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            # Listen for messages from the frontend (manual override slider)
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            if data.get("type") == "manual_override":
                async with state_lock:
                    manual_override["active"] = bool(data.get("active", False))
                    manual_override["value"] = float(data.get("value", 0.0))

            elif data.get("type") == "emergency_response":
                action = data.get("action")  # "cancel" (driver confirmed they're OK)
                async with state_lock:
                    if emergency_prompt["pending"] and action == "cancel":
                        emergency_prompt["pending"] = False
                        emergency_prompt["acknowledged"] = True
                        observability.log_agent_action(
                            "escalate", ["emergency_cancelled_by_driver"], {}
                        )
                        await broadcast({
                            "type": "emergency_cancelled",
                            "timestamp": time.time(),
                        })
    except WebSocketDisconnect:
        connected_clients.discard(websocket)


async def broadcast(payload: dict):
    dead = set()
    message = json.dumps(payload)
    for ws in connected_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    connected_clients.difference_update(dead)


# ---------------------------------------------------------------------------
# BLE sensor wiring
# ---------------------------------------------------------------------------
def handle_reading(reading: SensorReading):
    # Called from the BLE background task (same event loop, so this is safe
    # to call synchronously -- it just stashes state; broadcasting happens
    # in the main tick loop below).
    global _last_reading
    _last_reading = reading


def handle_status(msg: str):
    global latest_status_text
    latest_status_text = msg
    if "no ble heart rate device" in msg.lower():
        demo_mode["active"] = True


_last_reading: SensorReading = None


async def ble_task():
    sensor = BleHeartRateSensor(on_reading=handle_reading, on_status=handle_status)
    await sensor.run()


# ---------------------------------------------------------------------------
# Demo mode: fake data generator (used if no real sensor is found)
# ---------------------------------------------------------------------------
def generate_demo_reading() -> SensorReading:
    # Slowly wander the fake BPM so the dashboard has something to show
    demo_mode["bpm"] += random.uniform(-1.5, 1.5)
    demo_mode["bpm"] = max(55, min(140, demo_mode["bpm"]))
    bpm = int(demo_mode["bpm"])
    # Fake RR intervals consistent with this BPM, with some jitter
    base_rr = 60000.0 / bpm
    rr = [base_rr + random.uniform(-15, 15) for _ in range(2)]
    return SensorReading(bpm=bpm, rr_intervals_ms=rr)


# ---------------------------------------------------------------------------
# Main tick loop: pulls latest reading, runs it through the AI layer,
# broadcasts the result. Runs ~5x/second.
# ---------------------------------------------------------------------------
async def tick_loop():
    global _last_reading
    while True:
        await asyncio.sleep(0.2)

        async with state_lock:
            override_active = manual_override["active"]
            override_value = manual_override["value"]

        assessment = None
        guardrail_note = None

        if override_active:
            # Manual override bypasses sensor guardrails (it's not sensor data)
            # but the speed rate limiter still applies -- even a manual test
            # value shouldn't be able to snap the car's speed instantly.
            assessment = engine.manual_assessment(override_value)

        else:
            if demo_mode["active"]:
                reading = generate_demo_reading()
            elif _last_reading is not None:
                reading = _last_reading
            else:
                reading = None

            if reading is not None:
                check = guardrails.check_reading(reading)
                if check.ok:
                    assessment = engine.ingest(reading)
                else:
                    guardrail_note = check.reason
                    observability.log_guardrail_reject(check.reason, {"bpm": reading.bpm})

            # Stale-data fallback: if we haven't seen a *valid* reading
            # recently, don't keep acting on old state -- fall back safely.
            if guardrails.sensor_guardrail.is_stale():
                safe_speed, reason = guardrails.fallback_state()
                assessment = None
                await broadcast({
                    "type": "assessment",
                    "stress_score": 0,
                    "bpm": None,
                    "baseline_bpm": None,
                    "rmssd": None,
                    "target_speed": round(safe_speed, 2),
                    "intervening": False,
                    "status": reason,
                    "demo_mode": demo_mode["active"],
                    "manual_override": override_active,
                    "guardrail_note": reason,
                    "agent_state": "normal",
                    "agent_actions": [],
                    "timestamp": time.time(),
                })
                continue

        if assessment is not None:
            # Rate-limit the AI's requested speed for safety before it goes
            # anywhere (dashboard, or eventually an actuator).
            limited_speed = guardrails.apply_speed_limit(assessment.target_speed)

            # Agentic layer: reason over sustained stress, not just this instant.
            agent_decision = safety_agent.step(assessment.stress_score, limited_speed)
            if agent_decision.actions:
                observability.log_agent_action(
                    agent_decision.state.value, agent_decision.actions,
                    {"stress_score": assessment.stress_score},
                )

            # --- Emergency contact flow ---
            # Triggered only on ESCALATE, and only once per escalation episode.
            # "acknowledged" tracks whether the driver has already responded
            # (cancelled) or been auto-notified for THIS episode, so the prompt
            # doesn't immediately reopen on the next 200ms tick just because
            # "pending" was cleared. It only resets once stress actually drops
            # back down (agent leaves ESCALATE entirely -- see agent.py, which
            # only transitions ESCALATE -> NORMAL, never back down a step).
            async with state_lock:
                if (
                    agent_decision.state == AgentState.ESCALATE
                    and not emergency_prompt["pending"]
                    and not emergency_prompt["acknowledged"]
                ):
                    emergency_prompt["pending"] = True
                    emergency_prompt["prompted_at"] = time.time()
                    emergency_prompt["reason"] = agent_decision.reasoning
                    await broadcast({
                        "type": "emergency_prompt",
                        "countdown_seconds": EMERGENCY_COUNTDOWN_SECONDS,
                        "reason": agent_decision.reasoning,
                        "timestamp": time.time(),
                    })

                elif agent_decision.state in (AgentState.NORMAL, AgentState.WATCH):
                    # Stress has come back down -- clear any pending prompt
                    # without notifying anyone, and reset for the next episode.
                    if emergency_prompt["pending"] or emergency_prompt["acknowledged"]:
                        emergency_prompt["pending"] = False
                        emergency_prompt["acknowledged"] = False
                        await broadcast({"type": "emergency_cancelled", "timestamp": time.time()})

                if emergency_prompt["pending"]:
                    elapsed = time.time() - emergency_prompt["prompted_at"]
                    if elapsed >= EMERGENCY_COUNTDOWN_SECONDS:
                        # No response from the driver -- notify the emergency
                        # contact (never 911; see emergency.py).
                        result = emergency_notifier.notify(emergency_prompt["reason"])
                        emergency_prompt["pending"] = False
                        emergency_prompt["acknowledged"] = True
                        observability.log_agent_action(
                            "escalate",
                            ["emergency_contact_notified" if result.sent else "emergency_contact_notify_failed"],
                            {"mode": result.mode, "detail": result.detail},
                        )
                        await broadcast({
                            "type": "emergency_notified",
                            "sent": result.sent,
                            "mode": result.mode,
                            "detail": result.detail,
                            "timestamp": time.time(),
                        })

            payload = {
                "type": "assessment",
                "stress_score": round(assessment.stress_score, 1),
                "bpm": assessment.bpm,
                "simulated_vitals": assessment.simulated,
                "baseline_bpm": round(assessment.baseline_bpm, 1) if assessment.baseline_bpm else None,
                "rmssd": round(assessment.rmssd, 1) if assessment.rmssd else None,
                "target_speed": round(limited_speed, 2),
                "intervening": assessment.intervening,
                "status": latest_status_text,
                "demo_mode": demo_mode["active"],
                "manual_override": override_active,
                "guardrail_note": guardrail_note,
                "agent_state": agent_decision.state.value,
                "agent_actions": agent_decision.actions,
                "agent_reasoning": agent_decision.reasoning,
                "timestamp": time.time(),
            }
            observability.log_assessment(payload)
            await broadcast(payload)


@app.on_event("startup")
async def startup():
    asyncio.create_task(ble_task())
    asyncio.create_task(tick_loop())
