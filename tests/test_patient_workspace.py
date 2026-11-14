from __future__ import annotations

from datetime import datetime, timedelta

from database import db


def _create_patient_workspace_schema(path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(path))
    with db.get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL,
                status TEXT NOT NULL,
                specialization TEXT,
                hospital TEXT
            );
            CREATE TABLE patient_records (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                patient_gender TEXT,
                dob TEXT,
                address TEXT,
                notes TEXT
            );
            CREATE TABLE patient_profiles (
                user_id INTEGER PRIMARY KEY,
                patient_gender TEXT,
                dob TEXT,
                address TEXT,
                medical_history TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE patient_doctor_assignments (
                patient_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, doctor_id)
            );
            CREATE TABLE patient_appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                scheduled_at TEXT NOT NULL,
                appointment_type TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'scheduled',
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
            INSERT INTO users (id, username, email, phone, password, role, status)
            VALUES (1, 'patient', 'patient@example.test', '+15555550101', 'hash', 'patient', 'active');
            INSERT INTO users (id, username, email, phone, password, role, status, specialization)
            VALUES (2, 'doctor', 'doctor@example.test', '+15555550102', 'hash', 'doctor', 'approved', 'Radiology');
            INSERT INTO patient_records (id, user_id, patient_gender, dob, address, notes)
            VALUES (1, 1, 'Female', '1990-02-03', 'Existing address', 'Existing clinical record');
            INSERT INTO patient_doctor_assignments (patient_id, doctor_id, status)
            VALUES (1, 2, 'active');
            """
        )


def test_patient_profile_edits_do_not_overwrite_clinical_record(tmp_path, monkeypatch):
    _create_patient_workspace_schema(tmp_path / "patient-profile.db", monkeypatch)

    assert db.get_patient_profile(1)["medical_history"] == "Existing clinical record"
    assert db.update_patient_profile(
        1,
        "Other",
        "1991-04-05",
        "New address",
        "Patient-provided context",
    )

    profile = db.get_patient_profile(1)
    assert profile["patient_gender"] == "Other"
    assert profile["dob"] == "1991-04-05"
    assert profile["address"] == "New address"
    assert profile["medical_history"] == "Patient-provided context"
    with db.get_connection() as conn:
        clinical_record = conn.execute(
            "SELECT address, notes FROM patient_records WHERE id = 1"
        ).fetchone()
    assert tuple(clinical_record) == ("Existing address", "Existing clinical record")
    assert db.update_patient_profile(2, "Other", None, None, None) is False


def test_patient_appointments_require_relationship_and_cancel_only_own_booking(
    tmp_path, monkeypatch
):
    _create_patient_workspace_schema(tmp_path / "patient-appointments.db", monkeypatch)
    appointment_time = (datetime.now() + timedelta(days=2)).replace(
        hour=10, minute=30, second=0, microsecond=0
    )

    created, message, appointment_id = db.create_patient_appointment(
        1,
        2,
        appointment_time.isoformat(timespec="minutes"),
        "Video consultation",
        "Review my recent MRI.",
    )
    assert created is True
    assert message == "Consultation scheduled."
    assert appointment_id is not None
    assert db.get_patient_appointments(1)[0]["doctor_name"] == "doctor"
    assert db.get_doctor_appointments(2)[0]["patient_name"] == "patient"

    conflict_time = appointment_time + timedelta(minutes=15)
    conflict, conflict_message, _ = db.create_patient_appointment(
        1,
        2,
        conflict_time.isoformat(timespec="minutes"),
        "In-person consultation",
        "Follow-up",
    )
    assert conflict is False
    assert "no longer available" in conflict_message
    assert db.cancel_patient_appointment(2, appointment_id) is False
    assert db.cancel_patient_appointment(1, appointment_id) is True
    assert db.get_patient_appointments(1)[0]["status"] == "cancelled"


def test_patient_appointment_rejects_missing_relationship_and_invalid_input(
    tmp_path, monkeypatch
):
    _create_patient_workspace_schema(tmp_path / "patient-appointment-validation.db", monkeypatch)
    with db.get_connection() as conn:
        conn.execute("DELETE FROM patient_doctor_assignments")
    future = (datetime.now() + timedelta(days=1)).isoformat(timespec="minutes")

    created, message, appointment_id = db.create_patient_appointment(
        1,
        2,
        future,
        "Video consultation",
        "Review scan",
    )
    assert created is False
    assert appointment_id is None
    assert "active care relationship" in message

    created, message, _ = db.create_patient_appointment(
        1,
        2,
        future,
        "Unrecognized type",
        "Review scan",
    )
    assert created is False
    assert "valid consultation type" in message
