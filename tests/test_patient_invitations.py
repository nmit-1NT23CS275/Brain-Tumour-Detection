from __future__ import annotations

import pytest

from authentication.auth import can_doctor_scan_patient
from database import db


def _create_invitation_test_schema(path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(path))
    with db.get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                phone TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                role TEXT NOT NULL,
                status TEXT NOT NULL,
                is_verified INTEGER DEFAULT 0,
                address TEXT,
                specialization TEXT,
                hospital TEXT,
                years_experience INTEGER
            );
            CREATE TABLE patient_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                patient_name TEXT,
                patient_gender TEXT,
                dob TEXT,
                address TEXT,
                notes TEXT,
                created_by_doctor_id INTEGER
            );
            CREATE TABLE patient_doctor_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                status TEXT DEFAULT 'active',
                created_by TEXT DEFAULT 'patient',
                assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, doctor_id)
            );
            CREATE TABLE patient_invitations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                relationship_role TEXT NOT NULL,
                institution TEXT,
                consent_note TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                responded_at TIMESTAMP
            );
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL
            );
            CREATE TABLE clinical_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                note_content TEXT NOT NULL,
                recommendation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                resource TEXT,
                details_json TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO users (username, email, phone, password, role, status, is_verified)
            VALUES ('doctor', 'doctor@example.test', '+15555550101', 'hash', 'doctor', 'approved', 1);
            INSERT INTO users (username, email, phone, password, role, status, is_verified)
            VALUES ('patient', 'patient@example.test', '+15555550102', 'hash', 'patient', 'active', 1);
            """
        )


def test_patient_invitation_requires_explicit_acceptance_before_access(tmp_path, monkeypatch):
    _create_invitation_test_schema(tmp_path / "invitations.db", monkeypatch)

    created, message, token = db.create_patient_relationship_invitation(
        doctor_id=1,
        patient_id=2,
        relationship_role="Consulting specialist",
        institution="Neuro Clinic",
        consent_note="Please review my MRI scans.",
    )

    assert created is True
    assert "must accept" in message
    assert token
    assert can_doctor_scan_patient(1, 2)[0] is False
    invitation = db.get_patient_invitation(token)
    assert invitation["status"] == "pending"
    assert invitation["relationship_role"] == "Consulting specialist"
    with db.get_connection() as conn:
        conn.execute("INSERT INTO predictions (user_id) VALUES (2)")
    with pytest.raises(PermissionError, match="active, verified patient relationship"):
        db.add_clinical_note(1, 1, "Clinical observation")

    accepted, message = db.respond_to_patient_invitation(token, 2, accept=True)

    assert accepted is True
    assert "now enabled" in message
    assert can_doctor_scan_patient(1, 2)[0] is True
    note_id = db.add_clinical_note(1, 1, "Independent clinical observation", "Follow-up review")
    assert note_id == 1
    assert db.get_clinical_notes_for_prediction(1)[0]["recommendation"] == "Follow-up review"
    assert db.respond_to_patient_invitation(token, 2, accept=True)[0] is False


def test_new_patient_invitation_creates_unverified_account_without_scan_access(
    tmp_path, monkeypatch
):
    _create_invitation_test_schema(tmp_path / "new-patient.db", monkeypatch)

    created, message, patient_id, token = db.create_doctor_added_patient_invitation(
        doctor_id=1,
        full_name="New Patient",
        email="new.patient@example.test",
        phone="+15555550103",
        dob="1990-01-02",
        relationship_role="Primary doctor",
        institution="Neuro Clinic",
    )

    assert created is True
    assert "acceptance" in message
    assert patient_id is not None
    assert token
    patient = db.get_user_by_id(patient_id)
    assert patient["is_verified"] == 0
    assert can_doctor_scan_patient(1, patient_id)[0] is False
    assert db.respond_to_patient_invitation(token, patient_id, accept=True)[0] is True
    assert can_doctor_scan_patient(1, patient_id)[0] is True
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT token_hash, status FROM patient_invitations WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
    assert row["token_hash"] != token
    assert row["status"] == "accepted"


def test_expired_invitation_cannot_be_accepted(tmp_path, monkeypatch):
    _create_invitation_test_schema(tmp_path / "expired.db", monkeypatch)
    created, _, token = db.create_patient_relationship_invitation(1, 2)
    assert created is True
    with db.get_connection() as conn:
        conn.execute(
            "UPDATE patient_invitations SET expires_at = '2000-01-01 00:00:00'"
        )

    invitation = db.get_patient_invitation(token)

    assert invitation["status"] == "expired"
    assert db.respond_to_patient_invitation(token, 2, accept=True)[0] is False
    assert can_doctor_scan_patient(1, 2)[0] is False


def test_doctor_profile_updates_only_professional_contact_fields(tmp_path, monkeypatch):
    _create_invitation_test_schema(tmp_path / "profile.db", monkeypatch)

    assert db.update_doctor_profile(1, "Neuro Clinic", "Clinic address", 12) is True
    with db.get_connection() as conn:
        doctor = conn.execute(
            "SELECT hospital, address, years_experience FROM users WHERE id = 1"
        ).fetchone()

    assert tuple(doctor) == ("Neuro Clinic", "Clinic address", 12)
    patient_update = db.update_doctor_profile(2, "Other Clinic", None, 2)
    assert patient_update is False
    with pytest.raises(ValueError, match="cannot be negative"):
        db.update_doctor_profile(1, "Neuro Clinic", None, -1)
