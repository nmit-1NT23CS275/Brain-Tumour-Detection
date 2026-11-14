from __future__ import annotations

from authentication import auth


def test_password_reset_request_does_not_disclose_unknown_email(monkeypatch):
    class SuccessfulProvider:
        def send_otp(self, recipient, otp_code):
            return True, "sent"

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("SMTP_USERNAME", "smtp-user")
    monkeypatch.setenv("SMTP_PASSWORD", "smtp-password")
    monkeypatch.setattr(auth, "EmailProvider", SuccessfulProvider)
    monkeypatch.setattr(auth, "generate_otp", lambda: "123456")
    monkeypatch.setattr(auth, "store_otp", lambda *args, **kwargs: (True, "stored"))

    monkeypatch.setattr(auth, "get_user_by_identifier", lambda email: None)
    unknown_result = auth.request_password_reset("nobody@example.com")

    monkeypatch.setattr(
        auth,
        "get_user_by_identifier",
        lambda email: {"id": 7, "email": "known@example.com"},
    )
    known_result = auth.request_password_reset("known@example.com")

    assert unknown_result == known_result
    assert unknown_result == (
        True,
        "If an account exists for this email, a recovery code has been sent.",
    )


def test_password_reset_updates_hash_only_after_code_verification(monkeypatch):
    user = {"id": 42, "email": "patient@example.com"}
    updated = {}
    monkeypatch.setattr(auth, "get_user_by_identifier", lambda email: user)
    monkeypatch.setattr(
        auth,
        "verify_otp_detailed",
        lambda email, code: (code == "123456", "Invalid or expired recovery code."),
    )
    monkeypatch.setattr(
        auth,
        "update_user_password",
        lambda user_id, hashed: updated.update(user_id=user_id, hashed=hashed) or True,
    )
    monkeypatch.setattr(auth, "log_audit_event", lambda **kwargs: None)

    rejected, _ = auth.complete_password_reset(
        "patient@example.com", "000000", "SecurePass1!", "SecurePass1!"
    )
    assert rejected is False
    assert updated == {}

    completed, message = auth.complete_password_reset(
        "PATIENT@example.com", "123456", "SecurePass1!", "SecurePass1!"
    )

    assert completed is True
    assert "updated" in message
    assert updated["user_id"] == 42
    assert auth.verify_password("SecurePass1!", updated["hashed"])


def test_password_reset_rejects_mismatched_confirmation(monkeypatch):
    monkeypatch.setattr(
        auth,
        "get_user_by_identifier",
        lambda email: {"id": 42, "email": email},
    )
    verified = []
    monkeypatch.setattr(
        auth,
        "verify_otp_detailed",
        lambda *args: verified.append(args) or (True, "verified"),
    )

    completed, message = auth.complete_password_reset(
        "patient@example.com", "123456", "SecurePass1!", "DifferentPass2!"
    )

    assert completed is False
    assert message == "Passwords do not match."
    assert verified == []
