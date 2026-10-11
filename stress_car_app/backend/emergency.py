"""
Emergency Contact Notifier
---------------------------
Sends a notification to a DESIGNATED PERSONAL EMERGENCY CONTACT (e.g. a
family member) -- never to 911 or any emergency dispatch service. Calling
911 automatically is out of scope for this prototype on purpose: a
heuristic stress score isn't a validated medical trigger, and false
automated emergency calls carry real legal and safety consequences. See
the project README for the reasoning.

This module is pluggable across TWO independent channels, either or both
of which can be enabled:
  - SMS via Twilio
  - Email via SMTP

If neither is configured, it runs in SIMULATED mode -- it logs exactly
what would have been sent, so the full pipeline (agent escalation ->
confirmation countdown -> notification) is testable and demoable with
zero external setup or risk of accidentally contacting anyone.

To enable SMS, set:
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_FROM_NUMBER       (a phone number you own on Twilio)
    EMERGENCY_CONTACT_PHONE  (who gets texted, e.g. "+15551234567")
    (requires: pip install twilio)

To enable email, set:
    SMTP_HOST                (e.g. "smtp.gmail.com")
    SMTP_PORT                (e.g. 587 -- defaults to 587 if unset)
    SMTP_USERNAME
    SMTP_PASSWORD            (for Gmail, use an App Password, not your
                               normal password: myaccount.google.com/apppasswords)
    SMTP_FROM_EMAIL          (often same as SMTP_USERNAME)
    EMERGENCY_CONTACT_EMAIL  (who gets emailed)

EMERGENCY_CONTACT_NAME is used for both channels if set (defaults to
"your emergency contact" in the message body if omitted).
"""

import os
import smtplib
import time
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import List, Optional


@dataclass
class EmergencyContact:
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None


@dataclass
class ChannelResult:
    channel: str        # "sms" or "email"
    sent: bool
    detail: str


@dataclass
class NotificationResult:
    sent: bool                 # True if at least one channel succeeded
    mode: str                  # "live", "simulated", or "unconfigured"
    detail: str                # human-readable summary
    channels: List[ChannelResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class EmergencyNotifier:
    def __init__(self):
        self.contact = self._load_contact_from_env()

        # --- SMS (Twilio) setup ---
        self._twilio_client = None
        self._twilio_from = os.environ.get("TWILIO_FROM_NUMBER")
        sid = os.environ.get("TWILIO_ACCOUNT_SID")
        token = os.environ.get("TWILIO_AUTH_TOKEN")
        if sid and token and self._twilio_from and self.contact and self.contact.phone:
            try:
                from twilio.rest import Client  # optional dependency, imported lazily
                self._twilio_client = Client(sid, token)
            except ImportError:
                print("[emergency] twilio package not installed — SMS channel disabled. "
                      "Run: pip install twilio")

        # --- Email (SMTP) setup ---
        self._smtp_host = os.environ.get("SMTP_HOST")
        self._smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self._smtp_username = os.environ.get("SMTP_USERNAME")
        self._smtp_password = os.environ.get("SMTP_PASSWORD")
        self._smtp_from = os.environ.get("SMTP_FROM_EMAIL", self._smtp_username)
        self._email_enabled = bool(
            self._smtp_host and self._smtp_username and self._smtp_password
            and self.contact and self.contact.email
        )

    @staticmethod
    def _load_contact_from_env() -> Optional[EmergencyContact]:
        name = os.environ.get("EMERGENCY_CONTACT_NAME", "your emergency contact")
        phone = os.environ.get("EMERGENCY_CONTACT_PHONE")
        email = os.environ.get("EMERGENCY_CONTACT_EMAIL")
        if phone or email:
            return EmergencyContact(name=name, phone=phone, email=email)
        return None

    def is_configured_for_real_sms(self) -> bool:
        return self._twilio_client is not None

    def is_configured_for_real_email(self) -> bool:
        return self._email_enabled

    def _build_message(self, reason: str) -> str:
        return (
            f"Automated alert: sustained high driver stress detected. "
            f"Reason: {reason}. This is a message from a prototype "
            f"driver-monitoring system, not emergency services."
        )

    def _send_sms(self, message: str) -> ChannelResult:
        try:
            self._twilio_client.messages.create(
                body=message, from_=self._twilio_from, to=self.contact.phone,
            )
            return ChannelResult("sms", True, f"SMS sent to {self.contact.phone}")
        except Exception as e:
            return ChannelResult("sms", False, f"SMS send failed: {e}")

    def _send_email(self, message: str) -> ChannelResult:
        try:
            msg = EmailMessage()
            msg["Subject"] = "Driver stress alert (automated, not emergency services)"
            msg["From"] = self._smtp_from
            msg["To"] = self.contact.email
            msg.set_content(message)

            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self._smtp_username, self._smtp_password)
                server.send_message(msg)

            return ChannelResult("email", True, f"Email sent to {self.contact.email}")
        except Exception as e:
            return ChannelResult("email", False, f"Email send failed: {e}")

    def notify(self, reason: str) -> NotificationResult:
        message = self._build_message(reason)

        if not self.contact:
            return NotificationResult(
                sent=False,
                mode="unconfigured",
                detail="No emergency contact configured (set EMERGENCY_CONTACT_PHONE "
                       "and/or EMERGENCY_CONTACT_EMAIL environment variables).",
            )

        channels: List[ChannelResult] = []

        if self._twilio_client:
            channels.append(self._send_sms(message))
        if self._email_enabled:
            channels.append(self._send_email(message))

        if channels:
            any_sent = any(c.sent for c in channels)
            summary = "; ".join(f"{c.channel}: {'ok' if c.sent else 'failed'}" for c in channels)
            return NotificationResult(
                sent=any_sent,
                mode="live",
                detail=summary,
                channels=channels,
            )

        # Simulated mode -- safe default, no external calls made
        print(f"[emergency] SIMULATED notification to {self.contact.name}: {message}")
        return NotificationResult(
            sent=True,
            mode="simulated",
            detail=f"[SIMULATED] Would have notified {self.contact.name}"
                   f"{' (' + self.contact.phone + ')' if self.contact.phone else ''}"
                   f"{' <' + self.contact.email + '>' if self.contact.email else ''}",
            channels=[],
        )

