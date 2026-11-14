"""Pluggable OTP Delivery Provider Interface and Implementations.

Supports Demo (Development), Twilio (SMS), and SMTP (Email) providers.
"""

from __future__ import annotations

import abc
import logging
import os
import smtplib
from email.mime.text import MIMEText
from typing import Optional

from utils import configure_logging

LOGGER = configure_logging(__name__)


class OTPProvider(abc.ABC):
    """Abstract base class for pluggable OTP delivery providers."""

    @abc.abstractmethod
    def send_otp(self, recipient: str, otp_code: str) -> tuple[bool, str]:
        """Send OTP code to recipient (phone number or email address).

        Returns:
            tuple[bool, str]: (Success status, user-facing message/details)
        """
        pass

    def send_invitation(self, recipient: str, invitation_url: str) -> tuple[bool, str]:
        return False, "This provider does not support email invitations."

    def send_account_credentials(
        self, recipient: str, patient_name: str, username: str,
        temporary_password: str, doctor_name: str,
    ) -> tuple[bool, str]:
        return False, "This provider does not support account emails."


class DemoProvider(OTPProvider):
    """Demo OTP provider for development/testing environments."""

    def send_otp(self, recipient: str, otp_code: str) -> tuple[bool, str]:
        app_env = os.getenv("APP_ENV", "development").lower()
        if app_env == "production":
            LOGGER.error("DemoProvider invoked in production environment! Rejecting OTP delivery.")
            return False, "Demo OTP provider is disabled in production environment."

        LOGGER.info("[DEMO OTP] Recipient: %s | Code: %s", recipient, otp_code)
        return True, f"Demo OTP generated and logged for {recipient}."

    def send_invitation(self, recipient: str, invitation_url: str) -> tuple[bool, str]:
        if os.getenv("APP_ENV", "development").lower() == "production":
            LOGGER.error("Demo invitation provider invoked in production.")
            return False, "Demo invitations are disabled in production."
        LOGGER.info("[DEMO INVITATION] Recipient: %s | Link: %s", recipient, invitation_url)
        return True, "Demo invitation link logged to the application server."

    def send_account_credentials(
        self, recipient: str, patient_name: str, username: str,
        temporary_password: str, doctor_name: str,
    ) -> tuple[bool, str]:
        if os.getenv("APP_ENV", "development").lower() == "production":
            LOGGER.error("Demo account email provider invoked in production.")
            return False, "Demo account emails are disabled in production."
        LOGGER.info(
            "[DEMO ACCOUNT EMAIL] Recipient: %s | Username: %s | Temporary password: %s",
            recipient, username, temporary_password,
        )
        return True, "Demo account credentials logged to the application server."


class TwilioProvider(OTPProvider):
    """Twilio SMS OTP delivery provider."""

    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.getenv("TWILIO_PHONE_NUMBER", "")

    def send_otp(self, recipient: str, otp_code: str) -> tuple[bool, str]:
        if not (self.account_sid and self.auth_token and self.from_number):
            LOGGER.error("Twilio credentials missing in environment variables.")
            return False, "SMS service configuration error."

        try:
            from twilio.rest import Client

            client = Client(self.account_sid, self.auth_token)
            message = client.messages.create(
                body=f"Your NeuroScan AI verification code is: {otp_code}. Valid for 5 minutes.",
                from_=self.from_number,
                to=recipient,
            )
            LOGGER.info("Twilio SMS sent successfully. Message SID: %s", message.sid)
            return True, "OTP SMS sent successfully."
        except Exception as exc:
            LOGGER.error("Twilio SMS delivery failed: %s", exc)
            return False, f"SMS delivery failed: {exc}"


class EmailProvider(OTPProvider):
    """SMTP Email OTP delivery provider."""

    def __init__(self):
        self.host = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.port = int(os.getenv("SMTP_PORT", "587"))
        self.username = os.getenv("SMTP_USERNAME", "")
        self.password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("SMTP_FROM_EMAIL", self.username)

    def send_otp(self, recipient: str, otp_code: str) -> tuple[bool, str]:
        if not (self.username and self.password):
            LOGGER.error("SMTP credentials missing in environment variables.")
            return False, "Email service configuration error."

        try:
            msg = MIMEText(
                f"Your NeuroScan AI security verification code is: {otp_code}\n\n"
                f"This code will expire in 5 minutes. If you did not request this, please ignore."
            )
            msg["Subject"] = "NeuroScan AI - Verification Code"
            msg["From"] = self.from_email
            msg["To"] = recipient

            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, [recipient], msg.as_string())

            LOGGER.info("SMTP OTP email sent successfully to %s", recipient)
            return True, "OTP email sent successfully."
        except Exception as exc:
            LOGGER.error("SMTP OTP email delivery failed: %s", exc)
            return False, f"Email delivery failed: {exc}"

    def send_invitation(self, recipient: str, invitation_url: str) -> tuple[bool, str]:
        if not (self.username and self.password):
            LOGGER.error("SMTP credentials missing; patient invitation not sent.")
            return False, "Email service configuration error."

        try:
            msg = MIMEText(
                "A clinician invited you to connect on NeuroScan AI.\n\n"
                "Review the clinician and relationship details, then accept or decline "
                "the invitation. The clinician cannot access your patient scans unless "
                "you accept. The link expires in 7 days.\n\n"
                f"Review invitation: {invitation_url}\n\n"
                "If you did not expect this invitation, you can ignore this email."
            )
            msg["Subject"] = "NeuroScan AI - Patient relationship invitation"
            msg["From"] = self.from_email
            msg["To"] = recipient
            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, [recipient], msg.as_string())
            LOGGER.info("Patient invitation email sent to %s", recipient)
            return True, "Invitation email sent."
        except (smtplib.SMTPException, OSError, ValueError) as exc:
            LOGGER.error("Patient invitation delivery failed: %s", exc)
            return False, "Invitation email could not be delivered."

    def send_account_credentials(
        self, recipient: str, patient_name: str, username: str,
        temporary_password: str, doctor_name: str,
    ) -> tuple[bool, str]:
        if not (self.username and self.password):
            LOGGER.error("SMTP credentials missing; patient account email not sent.")
            return False, "Email service configuration error: SMTP credentials are missing."

        body = (
            f"Hello {patient_name},\n\n"
            "A NeuroScan AI patient account has been created for you by "
            f"Dr. {doctor_name}.\n\n"
            "ACCOUNT DETAILS\n"
            "---------------\n"
            f"Username: {username}\n"
            f"Temporary password: {temporary_password}\n"
            f"Sign in: {os.getenv('APP_BASE_URL', 'http://localhost:8501').rstrip('/')}/?page=login\n\n"
            "SECURITY NOTICE\n"
            "----------------\n"
            "This temporary password is your initial reference password. Sign in and "
            "change it immediately from your Profile page. Do not share it with anyone.\n\n"
            "If you did not expect this account, contact your care provider."
        )
        try:
            msg = MIMEText(body)
            msg["Subject"] = "NeuroScan AI - Your patient account details"
            msg["From"] = self.from_email
            msg["To"] = recipient
            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, [recipient], msg.as_string())
            LOGGER.info("Patient account email sent to %s", recipient)
            return True, "Patient account email sent."
        except (smtplib.SMTPException, OSError, ValueError) as exc:
            LOGGER.error("Patient account email delivery failed: %s", exc)
            return False, "Patient account email could not be delivered."


def get_otp_provider() -> OTPProvider:
    """Factory function resolving configured OTP provider based on environment variables."""
    app_env = os.getenv("APP_ENV", "development").lower()
    provider_name = os.getenv("OTP_PROVIDER", "demo" if app_env != "production" else "twilio").lower()

    if provider_name == "twilio":
        return TwilioProvider()
    elif provider_name == "email" or provider_name == "smtp":
        return EmailProvider()
    elif provider_name == "demo":
        if app_env == "production":
            LOGGER.warning("Demo OTP provider configured in production! Falling back to TwilioProvider.")
            return TwilioProvider()
        return DemoProvider()
    else:
        LOGGER.warning("Unknown OTP_PROVIDER '%s'. Defaulting to DemoProvider.", provider_name)
        return DemoProvider()
