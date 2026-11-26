"""NeuroScan AI Streamlit application."""

from __future__ import annotations

import base64
import html
import io
import json
import logging
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from authentication import (
    clear_session,
    confirm_otp,
    create_auth_token,
    current_user_id,
    current_username,
    current_user_role,
    current_user_status,
    hash_password,
    is_admin,
    is_doctor,
    is_patient,
    is_logged_in,
    login_user,
    register_user,
    send_otp,
    send_patient_account_credentials,
    send_patient_invitation,
    set_session,
    restore_session_from_token,
    save_uploaded_doctor_document,
    can_doctor_scan_patient,
    request_password_reset,
    complete_password_reset,
    validate_email,
    validate_password,
    validate_phone,
    verify_password,
)
from backend import analyze_uploaded_scan
from chatbot import (
    SUGGESTED_QUESTIONS,
    get_llm_client,
    get_response_dict as mastishka_response_dict,
)
from database import (
    create_patient_record,
    create_doctor_added_patient_account,
    get_dashboard_stats,
    get_user_predictions,
    get_patient_profile,
    update_patient_profile,
    initialize_database,
    save_prediction,
    get_all_doctors,
    update_doctor_verification,
    get_all_users,
    update_user_status,
    assign_doctor_to_patient,
    get_assigned_doctors_for_patient,
    create_patient_appointment,
    get_patient_appointments,
    get_doctor_appointments,
    cancel_patient_appointment,
    is_patient_assigned_to_doctor,
    add_clinical_note,
    get_clinical_notes_for_prediction,
    get_doctor_assigned_predictions,
    log_audit_event,
    get_audit_logs,
    get_prediction_by_id,
    get_user_by_id,
    get_doctor_documents,
    find_patient_by_identifier,
    create_doctor_added_patient_invitation,
    create_patient_relationship_invitation,
    get_patient_invitation,
    get_doctor_pending_invitations,
    respond_to_patient_invitation,
    revoke_patient_invitation,
    update_user_password,
    update_doctor_profile,
    get_doctor_assigned_patients_with_details,
)
from frontend import (
    inject_doctor_workspace_theme,
    inject_admin_workspace_theme,
    inject_patient_workspace_theme,
    inject_theme,
)
from model import CLASS_INFO, is_model_available
from preprocessing import is_probable_brain_mri, load_image_from_upload, plot_pixel_histogram, plot_preprocessing_steps, preprocess_array, validate_image
from reports import (
    generate_online_ai_report,
    generate_pdf_report,
    is_online_ai_report_configured,
)
from utils.web_search import is_web_search_available


matplotlib.use("Agg")
load_dotenv()

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()
initialize_database()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_online_ai_report(
    prediction: str,
    confidence: float,
    probabilities_json: str,
    base_summary: str,
) -> dict:
    """Cache source-grounded summaries by non-identifying model output only."""
    return generate_online_ai_report(
        prediction,
        confidence,
        json.loads(probabilities_json),
        base_summary,
    )


def _report_ai_fields(
    prediction: str,
    confidence: float,
    probabilities: dict[str, float],
    base_summary: str | None,
) -> dict:
    """Use online research when configured, otherwise disclose the local-only report."""
    summary = base_summary or "No model-generated summary is available."
    if not is_online_ai_report_configured():
        return {
            "summary": summary,
            "sources": [],
            "status": (
                "Online AI research is not configured; this report contains the local "
                "model summary only."
            ),
        }
    try:
        with st.spinner("Searching trusted medical sources and preparing the AI report..."):
            result = _cached_online_ai_report(
                prediction,
                confidence,
                json.dumps(probabilities, sort_keys=True),
                summary,
            )
    except RuntimeError:
        logging.getLogger(__name__).exception("Online AI report generation failed.")
        return {
            "summary": summary,
            "sources": [],
            "status": (
                "Online AI research could not be completed; this report contains the "
                "local model summary only. Check provider configuration and application logs."
            ),
        }
    return {
        "summary": result["summary"],
        "sources": result["sources"],
        "status": (
            "Educational context generated from online medical sources; not a diagnosis "
            "or a substitute for clinical interpretation."
        ),
    }

_BRAIN_ICON_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="M12 4.2c-1.1-1.6-3.8-1.8-5.1.1-2.4-.2-4 2.3-3 4.4-1.5 2-.4 4.8 1.7 5.2.4 2.2 2.8 3.1 4.4 1.9.4.9 1.1 1.4 2 1.5V4.2Z"/>'
    '<path d="M12 4.2c1.1-1.6 3.8-1.8 5.1.1 2.4-.2 4 2.3 3 4.4 1.5 2 .4 4.8-1.7 5.2-.4 2.2-2.8 3.1-4.4 1.9-.4.9-1.1 1.4-2 1.5V4.2Z"/>'
    '<path d="M8.2 7.2c1 .3 1.8 1.2 1.8 2.3M6.5 11.2c1 .1 1.7.6 2 1.4m7.3-5.4c-1 .3-1.8 1.2-1.8 2.3m3.5 1.7c-1 .1-1.7.6-2 1.4"/>'
    '</svg>'
)
_SIDEBAR_ICONS = {
    "dashboard": '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.5"/></svg>',
    "patients": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 20v-1.6a3.4 3.4 0 0 0-3.4-3.4H6.4A3.4 3.4 0 0 0 3 18.4V20"/><circle cx="9.5" cy="7.5" r="3.5"/><path d="M17 11a3.5 3.5 0 0 0 0-6.8M17 15h.6a3.4 3.4 0 0 1 3.4 3.4V20"/></svg>',
    "add_patient": '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="9" cy="7.5" r="3.5"/><path d="M2.5 20v-1.6A3.4 3.4 0 0 1 5.9 15h6.2a3.4 3.4 0 0 1 3.4 3.4V20M19 8v7m-3.5-3.5h7"/></svg>',
    "scan": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m13-5v3a2 2 0 0 1-2 2h-3"/><circle cx="12" cy="12" r="4"/></svg>',
    "reports": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 3.5h7l5 5V20a1.5 1.5 0 0 1-1.5 1.5h-10A1.5 1.5 0 0 1 6 20V5a1.5 1.5 0 0 1 1-1.5Z"/><path d="M14 3.5V9h5M9 13h6m-6 4h6"/></svg>',
    "training": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 5.5A2.5 2.5 0 0 1 5.5 3H12v17H5.5A2.5 2.5 0 0 0 3 22V5.5Zm18 0A2.5 2.5 0 0 0 18.5 3H12v17h6.5A2.5 2.5 0 0 1 21 22V5.5Z"/><path d="M6.5 7H9m8-0h-2.5"/></svg>',
    "brain": _BRAIN_ICON_SVG,
    "profile": '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="8" r="3.5"/><path d="M4.5 21a7.5 7.5 0 0 1 15 0"/></svg>',
    "upload": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4m-4.5 4.5L12 4l4.5 4.5M4 15.5v3A2.5 2.5 0 0 0 6.5 21h11a2.5 2.5 0 0 0 2.5-2.5v-3"/></svg>',
    "history": '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2"/></svg>',
    "consultation": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 11.5a7.5 7.5 0 0 1-8 7.5 8.5 8.5 0 0 1-3.5-.8L4 20l1.2-4A7.2 7.2 0 0 1 4 11.5 7.5 7.5 0 0 1 12 4a7.5 7.5 0 0 1 8.5 7.5Z"/><path d="M8.5 11.5h7m-7 3h4"/></svg>',
    "logout": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 17l5-5-5-5m5 5H3"/><path d="M12 3h6a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-6"/></svg>',
}


def _card_start():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)


def _card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def _metric_tile(column, value: str, label: str, color: str = "#16b9ff"):
    column.markdown(
        f"""
        <div class="metric-tile">
          <div class="metric-value" style="color:{color};">{value}</div>
          <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="app-card" style="padding:1.6rem 1.8rem;">
          <div class="hero-title">{title}</div>
          <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _doctor_display_name(user: dict | None) -> str:
    name = str(
        (user or {}).get("full_name")
        or (user or {}).get("username")
        or "Doctor"
    ).strip()
    if name.lower().startswith("dr."):
        name = name[3:].strip()
    elif name.lower().startswith("dr_"):
        name = name[3:]
    name = " ".join(name.replace("_", " ").split())
    return name.title() if name.islower() or name.isupper() else name


def _patient_display_name(user: dict | None) -> str:
    name = str(
        (user or {}).get("full_name")
        or (user or {}).get("username")
        or "Patient"
    ).strip()
    name = " ".join(name.replace("_", " ").split())
    return name.title() if name.islower() or name.isupper() else name


def _patient_workspace_header(section: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="patient-workspace-marker"></div>
        <div class="workspace-section">{html.escape(section)}</div>
        <h1>{html.escape(title)}</h1>
        <div class="workspace-subtitle">{html.escape(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


def _doctor_workspace_header(section: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="doctor-workspace-marker"></div>
        <div class="workspace-section">{html.escape(section)}</div>
        <h1>{html.escape(title)}</h1>
        <div class="workspace-subtitle">{html.escape(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


def _doctor_page_access() -> int | None:
    if not is_doctor() or current_user_status() not in ("active", "approved"):
        st.error("Access restricted: an approved doctor account is required.")
        return None
    return current_user_id()


def _render_workspace_nav_item(
    label: str,
    page_key: str,
    icon_name: str,
    active: bool,
    key_prefix: str,
) -> None:
    icon_column, button_column = st.columns([0.16, 1], gap="small")
    with icon_column:
        active_class = " is-active" if active else ""
        st.markdown(
            f'<span class="workspace-nav-icon{active_class}">'
            f'{_SIDEBAR_ICONS[icon_name]}</span>',
            unsafe_allow_html=True,
        )
    with button_column:
        if st.button(
            label,
            key=f"{key_prefix}_sidebar_{page_key}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state["page"] = page_key
            st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        if is_admin():
            st.markdown(
                f'<div class="doctor-sidebar-brand"><span class="doctor-sidebar-logo">{_BRAIN_ICON_SVG}</span>'
                '<strong>NeuroScan AI</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="doctor-sidebar-section-label">ADMIN WORKSPACE</div>',
                unsafe_allow_html=True,
            )
            admin_pages = {
                "Admin Console": ("admin_dashboard", "dashboard"),
                "Doctor Verification": ("admin_doctor_verification", "profile"),
                "User Management": ("admin_user_management", "patients"),
                "System Metrics": ("admin_system_metrics", "reports"),
                "Audit Logs": ("audit_logs", "history"),
                "Profile": ("profile", "profile"),
            }
            st.session_state.setdefault("page", "admin_dashboard")
            for label, (page_key, icon_name) in admin_pages.items():
                _render_workspace_nav_item(
                    label, page_key, icon_name,
                    st.session_state["page"] == page_key, "admin",
                )
            st.markdown(
                f'<div class="doctor-sidebar-user"><span class="doctor-brand-mark">'
                f'{html.escape(current_username()[:1].upper())}</span>'
                f'<span><strong>{html.escape(current_username())}</strong>'
                '<small>Administrator</small></span></div>',
                unsafe_allow_html=True,
            )
            if st.button("Logout", key="admin_sidebar_logout", use_container_width=True):
                user_id = current_user_id()
                log_audit_event(user_id, "user_logout", "auth")
                clear_session()
                if "auth" in st.query_params:
                    del st.query_params["auth"]
                st.session_state["page"] = "home"
                st.rerun()
            return
        if (
            is_doctor()
            and current_user_status() in ("active", "approved")
        ):
            st.markdown(
                f'<div class="doctor-sidebar-brand"><span class="doctor-sidebar-logo">{_BRAIN_ICON_SVG}</span>'
                '<strong>NeuroScan AI</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="doctor-sidebar-section-label">DOCTOR WORKSPACE</div>',
                unsafe_allow_html=True,
            )
            doctor_pages = {
                "Clinical Dashboard": "doctor_dashboard",
                "My Patients": "doctor_patients",
                "Add Patient": "doctor_add_patient",
                "Scan Patient": "doctor_scan_patient",
                "Reports": "doctor_reports",
                "Training Hub": "training",
                "Mastishka AI": "mastishka",
                "Profile": "profile",
            }
            icons = {
                "Clinical Dashboard": "dashboard",
                "My Patients": "patients",
                "Add Patient": "add_patient",
                "Scan Patient": "scan",
                "Reports": "reports",
                "Training Hub": "training",
                "Mastishka AI": "brain",
                "Profile": "profile",
            }
            st.session_state.setdefault("page", "home")
            for label, page_key in doctor_pages.items():
                active = st.session_state["page"] == page_key
                _render_workspace_nav_item(label, page_key, icons[label], active, "doctor")
            st.markdown(
                f'<div class="doctor-sidebar-user"><span class="doctor-brand-mark">'
                f'{html.escape(current_username()[:1].upper())}</span>'
                f'<span><strong>{html.escape(_doctor_display_name({"username": current_username()}))}'
                '</strong><small>Doctor account</small></span></div>',
                unsafe_allow_html=True,
            )
            if st.button("Logout", key="doctor_sidebar_logout", use_container_width=True):
                user_id = current_user_id()
                log_audit_event(user_id, "user_logout", "auth")
                clear_session()
                if "auth" in st.query_params:
                    del st.query_params["auth"]
                st.session_state["page"] = "home"
                st.rerun()
            return
        if is_doctor() and current_user_status() not in ("active", "approved"):
            st.markdown(
                f'<div class="doctor-sidebar-brand"><span class="doctor-sidebar-logo">{_BRAIN_ICON_SVG}</span>'
                '<strong>NeuroScan AI</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="doctor-sidebar-section-label">DOCTOR WORKSPACE</div>',
                unsafe_allow_html=True,
            )
            pending_pages = {
                "Dashboard": ("doctor_pending", "dashboard"),
                "Mathiksha AI": ("doctor_pending_mastishka", "brain"),
                "Profile": ("doctor_pending_profile", "profile"),
            }
            for label, (page_key, icon_name) in pending_pages.items():
                _render_workspace_nav_item(
                    label, page_key, icon_name,
                    st.session_state.get("page") == page_key, "pending_doctor",
                )
            st.markdown(
                '<div class="pending-sidebar-callout"><strong>Pending verification</strong>'
                '<small>Your workspace is limited until your account is approved.</small></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="doctor-sidebar-user"><span class="doctor-brand-mark">'
                f'{html.escape(current_username()[:1].upper())}</span>'
                f'<span><strong>{html.escape(_doctor_display_name({"username": current_username()}))}</strong>'
                '<small>Pending doctor</small></span></div>',
                unsafe_allow_html=True,
            )
            if st.button("Logout", key="pending_doctor_sidebar_logout", use_container_width=True):
                user_id = current_user_id()
                log_audit_event(user_id, "user_logout", "auth")
                clear_session()
                if "auth" in st.query_params:
                    del st.query_params["auth"]
                st.session_state["page"] = "home"
                st.rerun()
            return
        if is_patient() and current_user_status() == "active":
            st.markdown(
                f'<div class="patient-sidebar-brand"><span class="patient-sidebar-logo">{_BRAIN_ICON_SVG}</span>'
                '<strong>NeuroScan AI</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="patient-sidebar-section-label">PATIENT WORKSPACE</div>',
                unsafe_allow_html=True,
            )
            patient_pages = {
                "Dashboard": ("dashboard", "dashboard"),
                "Upload MRI": ("upload", "upload"),
                "Scan History": ("history", "history"),
                "Reports": ("patient_reports", "reports"),
                "My Doctors": ("patient_doctors", "patients"),
                "Consultation": ("consultation", "consultation"),
                "Mastishka AI": ("mastishka", "brain"),
                "Profile": ("profile", "profile"),
            }
            st.session_state.setdefault("page", "dashboard")
            for label, (page_key, icon_name) in patient_pages.items():
                active = st.session_state["page"] == page_key
                _render_workspace_nav_item(label, page_key, icon_name, active, "patient")
            st.markdown(
                '<a class="patient-support-link" href="mailto:support@neuroscan.ai">'
                'Help &amp; support</a>',
                unsafe_allow_html=True,
            )
            display_name = _patient_display_name({"username": current_username()})
            st.markdown(
                f'<div class="patient-sidebar-user"><span class="patient-brand-mark">'
                f'{html.escape(display_name[:1].upper())}</span>'
                f'<span><strong>{html.escape(display_name)}</strong>'
                '<small>Patient account</small></span></div>',
                unsafe_allow_html=True,
            )
            if st.button("Logout", key="patient_sidebar_logout", use_container_width=True):
                user_id = current_user_id()
                log_audit_event(user_id, "user_logout", "auth")
                clear_session()
                if "auth" in st.query_params:
                    del st.query_params["auth"]
                st.session_state["page"] = "home"
                st.rerun()
            return

        st.markdown(
            f'<div class="doctor-sidebar-brand">'
            f'<span class="doctor-sidebar-logo">{_BRAIN_ICON_SVG}</span>'
            '<strong>NeuroScan AI</strong></div>',
            unsafe_allow_html=True,
        )
        st.caption("Medical MRI intelligence platform")

        if is_logged_in():
            role = current_user_role()
            status = current_user_status()
            role_label = f" ({role.capitalize()})" if role else ""
            st.success(f"Signed in as {current_username()}{role_label}")

            if role == "admin":
                pages = {
                    "👤 My Profile": "profile",
                    "Admin Console": "admin_dashboard",
                    "Audit Logs": "audit_logs",
                    "Upload & Scan": "upload",
                    "System Dashboard": "dashboard",
                    "Scan History": "history",
                    "Training Hub": "training",
                    "🧠 Mastishka AI": "mastishka",
                    "About": "about",
                }
            elif role == "doctor":
                if status in ("active", "approved"):
                    pages = {
                        "CLINICAL WORKSPACE": "doctor_dashboard",
                        "Overview": "doctor_dashboard",
                        "My Patients": "doctor_patients",
                        "Add Patient": "doctor_add_patient",
                        "Scan Patient": "doctor_scan_patient",
                        "Reports": "doctor_reports",
                        "Training Hub": "training",
                        "Mastishka AI": "mastishka",
                        "My Profile": "profile",
                    }
                else:
                    pages = {
                        "👤 My Profile": "profile",
                        "Application Status": "doctor_pending",
                        "🧠 Mastishka AI": "mastishka",
                        "About": "about",
                    }
            else:
                pages = {"Home": "home", "👤 My Profile": "profile", "About": "about"}
        else:
            pages = {
                "Home": "home",
                "Login": "login",
                "Register": "register",
                "🧠 Mastishka AI": "mastishka",
                "About": "about",
            }

        st.session_state.setdefault("page", "home")
        for label, page_key in pages.items():
            if label == "CLINICAL WORKSPACE":
                st.caption(label)
                continue
            if st.button(label, use_container_width=True, type="primary" if st.session_state["page"] == page_key else "secondary"):
                st.session_state["page"] = page_key
                st.rerun()

        if is_logged_in() and st.button("Logout", use_container_width=True):
            user_id = current_user_id()
            log_audit_event(user_id, "user_logout", "auth")
            clear_session()
            if "auth" in st.query_params:
                del st.query_params["auth"]
            st.session_state["page"] = "home"
            st.rerun()


def page_home() -> None:
    brain_path = Path(__file__).parent / "frontend" / "brain_illustration.png"
    brain_image = base64.b64encode(brain_path.read_bytes()).decode("ascii")
    st.markdown(
        """
        <style>
        @import url('https://fonts.cdnfonts.com/css/open-sauce-one');
        body:has(.landing-page),
        body:has(.landing-page) [data-testid="stAppViewContainer"],
        body:has(.landing-page) [data-testid="stMain"] {
          background:#fff !important;
          color:#193d45 !important;
          font-family:'Open Sauce One',sans-serif !important;
        }
        body:has(.landing-page) [data-testid="stSidebar"],
        body:has(.landing-page) header[data-testid="stHeader"],
        body:has(.landing-page) footer {display:none !important;}
        body:has(.landing-page) section.main > div.block-container {
          max-width:none !important;
          padding:0 !important;
        }
        body:has(.landing-page) div[data-testid="stMarkdown"]:has(.landing-page) {
          margin:0 !important;
        }
        .landing-page {
          --ink:#193d45;
          --muted:rgba(25,61,69,.74);
          --teal:#117c76;
          --pale:#f4f8f8;
          --line:rgba(25,61,69,.14);
          color:var(--ink);
          font-family:'Open Sauce One',sans-serif;
          font-size:16px;
          line-height:1.6;
        }
        .landing-page * {box-sizing:border-box;}
        .landing-page a {color:inherit;text-decoration:none;}
        .landing-page h1,
        .landing-page h2,
        .landing-page h3 {color:#193d45 !important;}
        .site-nav {
          height:52px;
          border-top:3px solid #193d45;
          background:#fff;
        }
        .site-nav-inner {
          height:100%;
          max-width:1160px;
          margin:0 auto;
          padding:0 24px;
          display:flex;
          align-items:center;
          gap:26px;
        }
        .brand {
          display:flex;
          align-items:center;
          gap:7px;
          margin-right:auto;
          font-size:12px;
          font-weight:700;
          white-space:nowrap;
        }
        .brand-mark {
          width:17px;height:17px;border-radius:6px;
          display:grid;place-items:center;
          color:#138f86;background:#e4f4f2;
          font-size:12px;
        }
        .brand-mark svg {width:13px;height:13px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;}
        .site-nav-links {display:flex;align-items:center;gap:25px;}
        .site-nav-links a,.nav-login {
          font-size:10px;color:#365b62;white-space:nowrap;
        }
        .nav-login,.button {
          display:inline-flex;align-items:center;justify-content:center;
          border-radius:4px;padding:8px 12px;
          font-size:10px;font-weight:600;white-space:nowrap;
        }
        .nav-login {border:1px solid var(--line);padding:7px 12px;}
        .button-primary {background:var(--teal);color:white !important;}
        .button-outline {background:#fff;border:1px solid #dfe9e8;color:var(--teal) !important;}
        .landing-hero {
          background:#f5f9f9;
          min-height:234px;
          display:flex;
          align-items:center;
        }
        .hero-inner {
          width:100%;max-width:1160px;margin:0 auto;padding:24px;
          display:grid;grid-template-columns:1.2fr .8fr;gap:32px;align-items:center;
        }
        .eyebrow {
          color:#138f86;font-size:8px;letter-spacing:.035em;
          text-transform:uppercase;margin:0 0 11px;
        }
        .hero-copy h1 {
          max-width:440px;margin:0 0 9px;
          color:#203f46;font-size:22px;line-height:1.36;letter-spacing:-.025em;
          font-weight:700;
        }
        .hero-copy p {margin:0;color:#6b858b;font-size:10px;}
        .hero-actions {display:flex;gap:7px;margin-top:12px;}
        .hero-note {font-size:8px !important;margin-top:10px !important;}
        .hero-art {
          width:100%;max-width:410px;justify-self:end;
          background:#edf6f5;padding:8px;
          border:1px solid #e8f0ef;
        }
        .hero-art img {display:block;width:100%;height:auto;aspect-ratio:1.5;object-fit:cover;}
        .hero-art-caption {padding:7px 4px 2px;color:#1c4148;font-size:11px;font-weight:600;}
        .hero-art-caption span {display:block;color:#6c8589;font-size:8px;font-weight:400;margin-top:3px;}
        .section-inner {max-width:1160px;margin:0 auto;padding:22px 24px;}
        .section-heading {margin:0 0 5px;font-size:13px;line-height:1.4;font-weight:700;}
        .section-intro {margin:0;color:var(--muted);font-size:9px;}
        .workspace-grid {display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;}
        .workspace-card {
          border-radius:7px;background:#f5f9f9;padding:12px;
          min-height:102px;
        }
        .card-kicker {margin:0 0 5px;color:#138f86;font-size:7px;text-transform:uppercase;}
        .workspace-card h3 {margin:0 0 5px;font-size:11px;line-height:1.35;}
        .workspace-card p {margin:0 0 9px;color:var(--muted);font-size:8px;line-height:1.45;}
        .workspace-card .button {padding:6px 9px;font-size:8px;background:#fff;border:1px solid #e4eceb;color:var(--teal);}
        .steps-section {background:#f5f9f9;}
        .steps-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin-top:11px;}
        .step-number {margin:0 0 3px;color:#138f86;font-size:13px;font-weight:700;}
        .step h3 {margin:0 0 4px;font-size:9px;font-weight:700;}
        .step p {margin:0;color:var(--muted);font-size:7px;line-height:1.45;}
        .connection-grid {display:grid;grid-template-columns:1fr 1fr;gap:26px;align-items:start;}
        .connection-grid .section-heading {font-size:14px;}
        .connection-list {display:grid;gap:8px;}
        .connection-list h3 {margin:0 0 2px;font-size:8px;}
        .connection-list p {margin:0;color:var(--muted);font-size:7px;}
        .safety-section {background:#e5f5f3;}
        .safety-grid {display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:9px;}
        .safety-grid h3 {margin:0 0 3px;font-size:8px;}
        .safety-grid p {margin:0;color:var(--muted);font-size:7px;line-height:1.5;}
        .landing-cta {text-align:center;padding:18px 24px 12px;}
        .landing-cta h2 {margin:0 0 6px;font-size:13px;}
        .landing-cta p {margin:0 0 9px;color:var(--muted);font-size:8px;}
        .landing-cta .button {font-size:8px;padding:7px 10px;}
        @media (min-width:900px) {
          .site-nav {height:64px;}
          .landing-hero {min-height:320px;}
          .hero-inner {padding-top:40px;padding-bottom:40px;}
          .hero-copy h1 {font-size:36px;}
          .hero-copy p {font-size:14px;}
          .eyebrow {font-size:10px;}
          .hero-note,.hero-art-caption span {font-size:10px !important;}
          .hero-art-caption {font-size:16px;}
          .nav-login,.site-nav-links a,.button {font-size:12px;}
          .section-inner {padding-top:34px;padding-bottom:34px;}
          .section-heading {font-size:20px;}
          .section-intro {font-size:13px;}
          .workspace-card {padding:22px;min-height:160px;}
          .card-kicker {font-size:9px;}
          .workspace-card h3 {font-size:16px;}
          .workspace-card p {font-size:12px;}
          .step-number {font-size:20px;}
          .step h3,.connection-list h3,.safety-grid h3 {font-size:13px;}
          .step p,.connection-list p,.safety-grid p {font-size:10px;}
          .connection-grid .section-heading {font-size:21px;}
          .landing-cta {padding-top:34px;padding-bottom:24px;}
          .landing-cta h2 {font-size:22px;}
          .landing-cta p {font-size:13px;}
        }
        @media (max-width:520px) {
          .site-nav-inner {gap:8px;padding:0 14px;}
          .site-nav-links {gap:9px;}
          .site-nav-links a {font-size:8px;}
          .brand {font-size:10px;}
          .nav-login,.site-nav .button {font-size:8px;padding:6px 7px;}
          .hero-inner {grid-template-columns:1.2fr .8fr;gap:12px;padding:18px 16px;}
          .hero-copy h1 {font-size:19px;}
          .hero-copy p {font-size:9px;}
          .hero-art {padding:5px;}
          .hero-art-caption {font-size:9px;}
          .section-inner {padding:19px 16px;}
          .workspace-grid {gap:8px;}
          .workspace-card {padding:10px;min-height:105px;}
          .steps-grid {gap:12px;}
          .connection-grid,.safety-grid {gap:14px;}
        }
        @media (max-width:380px) {
          .site-nav-links a:nth-child(3) {display:none;}
          .hero-inner {grid-template-columns:1fr;}
          .hero-copy h1 {max-width:290px;}
        }
        .landing-page,.landing-page * {font-family:'Open Sauce One',sans-serif !important;}
        .landing-page h1,.landing-page h2,.landing-page h3 {color:var(--ink) !important;}
        .landing-page p,.section-intro,.hero-copy p,.hero-art-caption span,
        .workspace-card p,.step p,.connection-list p,.safety-grid p,.landing-cta p {
          color:var(--muted) !important;
        }
        .site-nav,.landing-cta {background:#fff;}
        .landing-hero,.steps-section,.workspace-card,.safety-section {background:var(--pale) !important;}
        .site-nav {border-top:0;border-bottom:1px solid var(--line);}
        .brand-mark {color:var(--teal);background:rgba(17,124,118,.1);}
        .nav-login,.button-outline,.workspace-card .button {
          border-color:var(--line) !important;color:var(--teal) !important;
        }
        .button-primary {background:var(--teal) !important;color:#fff !important;}
        .brand,.site-nav-links a,.nav-login {font-size:16px !important;color:var(--ink) !important;}
        .hero-art-caption,.step-number {color:var(--ink) !important;}
        .hero-copy h1 {font-size:40px;color:var(--ink) !important;}
        .hero-copy p,.hero-note,.hero-art-caption span,.section-intro,
        .workspace-card p,.step p,.connection-list p,.safety-grid p,.landing-cta p {
          font-size:15px !important;line-height:1.6;
        }
        .eyebrow,.card-kicker {font-size:13px !important;color:var(--teal) !important;}
        .hero-art-caption {font-size:18px;}
        .section-heading,.connection-grid .section-heading {font-size:24px;}
        .workspace-card h3,.step h3,.connection-list h3,.safety-grid h3 {
          font-size:18px !important;line-height:1.4;
        }
        .step-number {font-size:22px;color:var(--teal) !important;}
        .button,.nav-login {font-size:15px !important;padding:11px 16px;}
        .hero-inner,.section-inner {max-width:1240px;}
        .section-inner {padding-top:42px;padding-bottom:42px;}
        .workspace-grid {gap:18px;margin-top:20px;}
        .workspace-card {padding:24px;border:1px solid var(--line);border-radius:16px;}
        .steps-grid {gap:26px;margin-top:20px;}
        .connection-grid,.safety-grid {gap:32px;}
        .hero-art {border-color:var(--line);background:#fff;border-radius:20px;padding:12px;}
        .hero-art img {border-radius:12px;}
        .hero-actions {gap:12px;margin-top:22px;flex-wrap:wrap;}
        @media(max-width:720px) {
          .site-nav {height:auto;}
          .site-nav-inner {height:auto;flex-wrap:wrap;gap:12px;padding:14px 18px;}
          .site-nav-links {gap:14px;flex-wrap:wrap;}
          .site-nav-links a,.nav-login {font-size:14px !important;}
          .brand {font-size:18px !important;}
          .hero-inner {grid-template-columns:1fr;gap:22px;padding:32px 20px;}
          .hero-copy h1 {font-size:32px;}
          .hero-art {max-width:520px;justify-self:start;}
          .workspace-grid,.steps-grid,.connection-grid,.safety-grid {grid-template-columns:1fr;}
          .section-inner {padding:30px 20px;}
          .landing-cta {padding:28px 20px;}
        }
        </style>
        <main class="landing-page">
          <nav class="site-nav" aria-label="Main navigation">
            <div class="site-nav-inner">
              <a class="brand" href="#top"><span class="brand-mark">__BRAIN_ICON_SVG__</span>NeuroScan AI</a>
              <div class="site-nav-links">
                <a href="#how-it-works">How it works</a>
                <a href="#for-patients">For patients</a>
                <a href="#for-doctors">For doctors</a>
              </div>
              <a class="nav-login" href="?page=login">Log in</a>
              <a class="button button-primary" href="?page=register">Create account</a>
            </div>
          </nav>
          <section class="landing-hero" id="top">
            <div class="hero-inner">
              <div class="hero-copy">
                <div class="eyebrow">NeuroScan AI · Intelligence, with care</div>
                <h1>A clearer view.<br>A more connected<br>care journey.</h1>
                <p>Bring brain MRI insights, clinical review and your care team together in one place.</p>
                <div class="hero-actions">
                  <a class="button button-primary" href="?page=register">Create your account</a>
                  <a class="button button-outline" href="?page=login">Log in</a>
                </div>
                <p class="hero-note">For patients and verified healthcare professionals.</p>
              </div>
              <div class="hero-art">
                <img src="data:image/png;base64,BRAIN_IMAGE" alt="Illustration of a brain scan in a softly glowing glass sphere">
                <div class="hero-art-caption">Insight is only the beginning.<span>AI supports your clinicians. Your care stays human.</span></div>
              </div>
            </div>
          </section>
          <section id="how-it-works">
            <div class="section-inner">
              <h2 class="section-heading">One platform. Your own care workspace.</h2>
              <p class="section-intro">Start with the role that fits your care journey.</p>
              <div class="workspace-grid">
                <article class="workspace-card" id="for-patients">
                  <div class="card-kicker">For patients</div>
                  <h3>Understand your next step</h3>
                  <p>Upload your brain MRI, follow your reports and stay connected with your doctor.</p>
                  <a class="button" href="?page=register">Sign up as a patient</a>
                </article>
                <article class="workspace-card" id="for-doctors">
                  <div class="card-kicker">For doctors</div>
                  <h3>Keep clinical care connected</h3>
                  <p>Review patient scans, manage your patients and bring clinical context to AI insights.</p>
                  <a class="button" href="?page=register">Register as a doctor</a>
                </article>
              </div>
            </div>
          </section>
          <section class="steps-section">
            <div class="section-inner">
              <h2 class="section-heading">From scan to a more informed conversation.</h2>
              <div class="steps-grid">
                <article class="step"><div class="step-number">01</div><h3>Create your account</h3><p>Choose Patient or Doctor and set up your profile. Doctor access is subject to verification.</p></article>
                <article class="step"><div class="step-number">02</div><h3>Bring your care together</h3><p>Patients can upload MRI scans and follow reports. Doctors can review their assigned patients.</p></article>
                <article class="step"><div class="step-number">03</div><h3>Review with your clinician</h3><p>Use your report to support a conversation with your care team, not to replace it.</p></article>
              </div>
            </div>
          </section>
          <section>
            <div class="section-inner connection-grid">
              <div>
                <h2 class="section-heading">Less scattered.<br>More connected.</h2>
                <p class="section-intro">Keep the important parts of your brain health journey within reach.</p>
              </div>
              <div class="connection-list">
                <article><h3>Your scan history</h3><p>Return to your uploads and reports from your patient workspace.</p></article>
                <article><h3>Your care team</h3><p>Stay connected to your doctors and consultation details.</p></article>
                <article><h3>A dedicated clinical view</h3><p>Give verified doctors a focused workspace for patient review.</p></article>
              </div>
            </div>
          </section>
          <section class="safety-section">
            <div class="section-inner">
              <h2 class="section-heading">Designed to support care. Never to replace it.</h2>
              <div class="safety-grid">
                <article><h3>Clinician-led decisions</h3><p>AI-generated insights are decision support, not a diagnosis. Discuss findings with a qualified healthcare professional.</p></article>
                <article><h3>Role-based access</h3><p>Your approved account determines your workspace. Doctor registration requires verification; administrator accounts are invitation-only.</p></article>
              </div>
            </div>
          </section>
          <section class="landing-cta">
            <h2>Take the next step in your care journey.</h2>
            <p>Create an account, or log in to your existing workspace.</p>
            <a class="button button-primary" href="?page=register">Create account</a>
          </section>
        </main>
        """.replace("BRAIN_IMAGE", brain_image).replace("__BRAIN_ICON_SVG__", _BRAIN_ICON_SVG),
        unsafe_allow_html=True,
    )


def _render_auth_frame(eyebrow: str, heading: str, description: str):
    brain_path = Path(__file__).parent / "frontend" / "brain_illustration.png"
    brain_image = base64.b64encode(brain_path.read_bytes()).decode("ascii")
    st.markdown(
        """
        <style>
        @import url('https://fonts.cdnfonts.com/css/open-sauce-one');
        body:has(.auth-page),
        body:has(.auth-page) [data-testid="stAppViewContainer"],
        body:has(.auth-page) [data-testid="stMain"] {
          background:#fff !important;color:#193d45 !important;
          font-family:'Open Sauce One',sans-serif !important;
        }
        body:has(.auth-page) [data-testid="stSidebar"],
        body:has(.auth-page) header[data-testid="stHeader"],
        body:has(.auth-page) footer {display:none !important;}
        body:has(.auth-page) section.main > div.block-container {
          max-width:1100px !important;padding:0 24px 36px !important;
        }
        .auth-page {--ink:#193d45;--muted:rgba(25,61,69,.74);--teal:#117c76;color:var(--ink);font-family:'Open Sauce One',sans-serif;}
        .auth-page * {font-family:'Open Sauce One',sans-serif !important;}
        .auth-page h1,.auth-page h2,.auth-page h3 {color:#193d45 !important;}
        .auth-nav {height:78px;display:flex;align-items:center;gap:18px;border-bottom:1px solid rgba(25,61,69,.14);}
        .auth-brand {display:flex;align-items:center;gap:12px;margin-right:auto;font-size:20px;font-weight:700;color:#193d45 !important;text-decoration:none;}
        .auth-brand-mark {display:grid;place-items:center;width:38px;height:38px;border-radius:12px;color:#117c76;background:rgba(17,124,118,.1);}
        .auth-brand-mark svg {width:25px;height:25px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;}
        .auth-nav-link {font-size:15px;color:#117c76 !important;text-decoration:none;}
        body:has(.auth-page) [data-testid="stHorizontalBlock"] {gap:48px !important;align-items:center;}
        .auth-aside {
          background:#f4f8f8;border:1px solid rgba(25,61,69,.14);border-radius:24px;
          padding:34px;min-height:0;margin-top:12px;
        }
        .auth-aside-kicker {color:#117c76;font-size:13px;font-weight:700;text-transform:uppercase;margin-bottom:14px;}
        .auth-aside h1 {font-size:32px;line-height:1.25;margin:0 0 14px;font-weight:700;color:#193d45 !important;}
        .auth-aside-copy {color:rgba(25,61,69,.74);font-size:16px;line-height:1.6;margin:0 0 22px;}
        .auth-art {width:100%;height:auto;aspect-ratio:2.1;object-fit:cover;border-radius:16px;display:block;}
        .auth-aside-features {display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px;}
        .auth-aside-features strong {display:block;font-size:15px;margin-bottom:4px;}
        .auth-aside-features span {display:block;color:rgba(25,61,69,.74);font-size:14px;}
        .auth-content {padding:8px 0 0;}
        .auth-content h2 {font-size:30px;margin:0 0 6px;color:#193d45 !important;}
        .auth-content-intro {color:rgba(25,61,69,.74);font-size:16px;margin:0 0 22px;}
        body:has(.auth-page) [data-testid="stWidgetLabel"] p {font-size:15px;color:#193d45;font-weight:600;}
        body:has(.auth-page) [data-testid="stTextInput"] div[data-baseweb="input"] {
          min-height:48px;background:#161b22 !important;border:1px solid rgba(17,124,118,.72) !important;
          border-radius:10px !important;box-shadow:none !important;
        }
        body:has(.auth-page) [data-testid="stTextInput"] input {
          min-height:48px;border:0 !important;border-radius:0 !important;box-shadow:none !important;
          background:transparent !important;color:#fff !important;font-size:15px;
          -webkit-text-fill-color:#fff !important;color-scheme:dark;
        }
        body:has(.auth-page) [data-testid="stTextInput"] input::placeholder {color:rgba(255,255,255,.72) !important;opacity:1;}
        body:has(.auth-page) [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
          border-color:#117c76 !important;box-shadow:0 0 0 3px rgba(17,124,118,.13) !important;
        }
        body:has(.auth-page) [data-testid="stTextInput"] div[data-baseweb="input"] button {
          background:transparent !important;color:#fff !important;
          border:0 !important;box-shadow:none !important;
        }
        body:has(.auth-page) [data-testid="InputInstructions"] {display:none !important;}
        body:has(.auth-page) [data-testid="stFormSubmitButton"] button {
          min-height:48px;border:0;border-radius:10px;background:#117c76;color:#fff;
          font-size:15px;font-weight:700;width:100%;
        }
        body:has(.auth-page) [data-testid="stFormSubmitButton"] button:hover {background:#193d45;color:#fff;}
        body:has(.auth-page) [data-testid="stRadio"] > label {font-size:15px;font-weight:600;}
        body:has(.auth-page) [data-testid="stRadio"] div[role="radiogroup"] {gap:8px;}
        body:has(.auth-page) [data-testid="stRadio"] div[role="radiogroup"] > label {
          flex:1;border:1px solid rgba(25,61,69,.14);border-radius:10px;padding:12px 14px;
          background:#fff;align-items:flex-start;
        }
        body:has(.auth-page) [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
          border-color:#117c76;background:rgba(17,124,118,.08);
        }
        body:has(.auth-page) [data-testid="stRadio"] div[role="radiogroup"] > label p {font-size:15px;font-weight:600;color:#193d45 !important;}
        body:has(.auth-page) [data-testid="stCaptionContainer"] p {font-size:13px;color:rgba(25,61,69,.74);}
        .auth-inline {text-align:center;font-size:14px;color:#193d45;padding:12px 0;}
        .auth-inline a {color:#117c76;font-weight:700;text-decoration:none;}
        .auth-note {background:#f4f8f8;border-radius:12px;padding:14px 16px;color:#193d45;font-size:14px;margin-top:14px;}
        .auth-note p {margin:4px 0 0;color:rgba(25,61,69,.74);font-size:13px;}
        @media(max-width:720px) {
          body:has(.auth-page) section.main > div.block-container {padding:0 16px 24px !important;}
          body:has(.auth-page) [data-testid="stHorizontalBlock"] {gap:18px !important;}
          body:has(.auth-page) [data-testid="stTextInput"] input {font-size:14px;}
          .auth-aside {padding:24px 20px;min-height:0;}
          .auth-aside h1 {font-size:26px;}
        }
        @media(max-width:560px) {
          body:has(.auth-page) [data-testid="stHorizontalBlock"] {flex-direction:column !important;gap:8px !important;}
          body:has(.auth-page) [data-testid="stHorizontalBlock"] [data-testid="column"] {width:100% !important;flex:1 1 100% !important;}
          .auth-aside {padding:20px 16px;}
          .auth-art {max-height:180px;}
          .auth-content {padding-top:12px;}
          .auth-content h2 {font-size:26px;}
        }
        </style>
        <main class="auth-page">
          <nav class="auth-nav" aria-label="Account navigation">
            <a class="auth-brand" href="?page=home"><span class="auth-brand-mark">__BRAIN_ICON_SVG__</span>NeuroScan AI</a>
            <a class="auth-nav-link" href="?page=home">Home</a> <a class="auth-nav-link" href="?page=login">Log in</a>
          </nav>
        </main>
        """.replace("__BRAIN_ICON_SVG__", _BRAIN_ICON_SVG),
        unsafe_allow_html=True,
    )
    return f"""
        <section class="auth-aside">
          <div class="auth-aside-kicker">{eyebrow}</div>
          <h1>{heading}</h1>
          <p class="auth-aside-copy">{description}</p>
          <img class="auth-art" src="data:image/png;base64,{brain_image}" alt="Softly glowing brain scan illustration">
        </section>
    """


def page_register() -> None:
    aside_html = _render_auth_frame(
        "Intelligence, with care",
        "Your care journey starts here.",
        "Choose your role to get the right workspace for your scans, your patients and your care team.",
    )
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown(aside_html, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="auth-content"><h2>Create your account</h2><p class="auth-content-intro">Start your NeuroScan AI care journey.</p></div>', unsafe_allow_html=True)
        st.caption("I am signing up as")
        account_type = st.radio(
            "I am signing up as",
            ["Patient", "Doctor"],
            horizontal=True,
            label_visibility="collapsed",
            key="register_account_type",
        )
        role = "doctor" if account_type == "Doctor" else "patient"
        if role == "doctor":
            st.caption("Clinical review and patient workspace. Doctor accounts require verification.")
        else:
            st.caption("My scans and care team.")
        with st.form("register_form"):
            first_name, last_name = st.columns(2)
            with first_name:
                first = st.text_input("First name", placeholder="Enter your first name")
            with last_name:
                last = st.text_input("Last name", placeholder="Enter your last name")
            email = st.text_input("Email address", placeholder="you@example.com")
            phone = st.text_input("Phone number", placeholder="Include country code, e.g. +1 555 123 4567")
            password = st.text_input("Password", type="password", placeholder="Create a strong password")

            specialization = None
            qualification = None
            license_no = None
            hospital = None
            years_exp = None
            address = None
            deg_file = None
            lic_file = None
            id_file = None

            if role == "doctor":
                st.markdown("#### Professional details")
                specialization = st.text_input("Medical specialization", placeholder="e.g. Neuro-oncology, Radiology")
                qualification = st.text_input("Medical qualification", placeholder="e.g. MBBS, MD")
                license_no = st.text_input("Medical license / registration number")
                hospital = st.text_input("Hospital / clinic affiliation")
                years_exp = int(st.number_input("Years of experience", min_value=0, max_value=60, value=0))
                address = st.text_area("Professional / clinic address")
                st.caption("Upload your degree and medical license for verification (PDF, PNG or JPG; max 10 MB each).")
                deg_file = st.file_uploader("Medical degree certificate *", type=["pdf", "png", "jpg", "jpeg"], key="deg_file")
                lic_file = st.file_uploader("Medical registration certificate / license *", type=["pdf", "png", "jpg", "jpeg"], key="lic_file")
                id_file = st.file_uploader("Professional / government ID (optional)", type=["pdf", "png", "jpg", "jpeg"], key="id_file")

            accepted_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy.")
            st.caption("Use at least 8 characters, including a number and a symbol.")
            submitted = st.form_submit_button("Create account")

        if submitted:
            if not accepted_terms:
                st.error("Please agree to the Terms of Service and Privacy Policy to continue.")
            elif not first.strip() or not last.strip():
                st.error("Please enter your first and last name.")
            elif role == "doctor" and (not deg_file or not lic_file):
                st.error("A medical degree certificate and medical license are required for doctor verification.")
            else:
                username = f"{first.strip()} {last.strip()}"
                success, message, user_id = register_user(
                    username=username,
                    email=email,
                    phone=phone,
                    password=password,
                    confirm=password,
                    role=role,
                    specialization=specialization,
                    qualification=qualification,
                    license_no=license_no,
                    hospital=hospital,
                    years_experience=years_exp,
                    address=address,
                )
                if success and user_id:
                    st.success(message)
                    if role == "doctor":
                        save_uploaded_doctor_document(deg_file, user_id, "degree")
                        save_uploaded_doctor_document(lic_file, user_id, "license")
                        if id_file:
                            save_uploaded_doctor_document(id_file, user_id, "gov_id")
                        st.success("Your application is pending administrative verification.")
                    else:
                        st.info("Your patient account is ready. You can sign in immediately.")
                else:
                    st.error(message)
        st.markdown('<div class="auth-inline">Already have an account? <a href="?page=login">Log in</a></div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-note"><strong>Administrator access is invitation-only.</strong><p>Use your organization’s invitation or contact your administrator.</p></div>', unsafe_allow_html=True)
        st.caption("Account creation is active. Your information is handled according to the Privacy Policy.")


def page_login() -> None:
    aside_html = _render_auth_frame(
        "Intelligence, with care",
        "A clearer view.<br>A more connected<br>care journey.",
        "Brain MRI insights, thoughtful clinical review and your care team — together in one secure place.",
    )
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown(aside_html, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="auth-content"><h2>Welcome back</h2><p class="auth-content-intro">Sign in to your NeuroScan AI account.</p></div>', unsafe_allow_html=True)
        with st.form("login_form"):
            identifier = st.text_input(
                "Email address",
                value=st.session_state.get("login_identifier_fill", ""),
                placeholder="name@example.com",
            )
            password = st.text_input(
                "Password",
                type="password",
                value=st.session_state.get("login_password_fill", ""),
                placeholder="Enter your password",
            )
            submitted = st.form_submit_button("Sign in")

        st.markdown('<div class="auth-inline"><a href="?page=forgot_password">Forgot password?</a></div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-inline">New to NeuroScan AI? <a href="?page=register">Create an account</a></div>', unsafe_allow_html=True)
        if submitted:
            st.session_state.pop("login_identifier_fill", None)
            st.session_state.pop("login_password_fill", None)
            success, message, user = login_user(identifier, password)
            if success and user:
                set_session(user)
                st.query_params["auth"] = create_auth_token(user)
                role = user.get("role", "patient")
                status = user.get("status", "active")

                if role == "admin":
                    st.session_state["page"] = "admin_dashboard"
                elif role == "doctor":
                    st.session_state["page"] = (
                        "doctor_dashboard"
                        if status in ("active", "approved")
                        else "doctor_pending"
                    )
                else:
                    st.session_state["page"] = "dashboard"

                st.success(message)
                st.rerun()
            else:
                st.error(message)

    # CRITICAL SECURITY NOTE: DEMO_MODE renders automated login buttons with demo credentials.
    # THIS MUST REMAIN DISABLED (DEMO_MODE=false) FOR ANY REAL DEPLOYMENT TO PREVENT UNAUTHORIZED ACCESS.
    import os
    demo_mode = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")

    if demo_mode:
        demo_patient_email = os.getenv("DEMO_PATIENT_EMAIL", "patient.demo@example.com")
        demo_patient_password = os.getenv("DEMO_PATIENT_PASSWORD", "Patient@123")
        demo_doctor_email = os.getenv("DEMO_DOCTOR_EMAIL", "doctor.demo@example.com")
        demo_doctor_password = os.getenv("DEMO_DOCTOR_PASSWORD", "Doctor@123")
        demo_pending_doctor_email = os.getenv("DEMO_PENDING_DOCTOR_EMAIL", "pending.doctor@example.com")
        demo_pending_doctor_password = os.getenv("DEMO_PENDING_DOCTOR_PASSWORD", "Doctor@123")
        demo_admin_email = os.getenv("DEMO_ADMIN_EMAIL", "admin.demo@example.com")
        demo_admin_password = os.getenv("DEMO_ADMIN_PASSWORD", "Admin@123")

        _card_start()
        st.subheader("🔑 Demonstration Credentials & Quick Login")
        st.caption("Faculty & Evaluator Demo Environment — One-Click Authentication into standard SQLite RBAC accounts.")

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)

        with dcol1:
            st.markdown("#### 👤 Demo Patient")
            st.write(f"**Email**: `{demo_patient_email}`  \n**Data**: 1 Sample Scan, Assigned Specialist")
            if st.button("⚡ Login as Demo Patient", use_container_width=True, type="secondary"):
                success, message, user = login_user(demo_patient_email, demo_patient_password)
                if success and user:
                    set_session(user)
                    st.session_state["page"] = "dashboard"
                    st.success("Authenticated as Demo Patient!")
                    st.rerun()
                else:
                    st.error(message)

        with dcol2:
            st.markdown("#### 👨‍⚕️ Verified Doctor — ✓ Approved")
            st.write(f"**Email**: `{demo_doctor_email}`  \n**Data**: Medical License, My Patients, Scan Workspace")
            if st.button("⚡ Login as Verified Doctor", use_container_width=True, type="primary"):
                success, message, user = login_user(demo_doctor_email, demo_doctor_password)
                if success and user:
                    set_session(user)
                    st.session_state["page"] = "doctor_dashboard"
                    st.success("Authenticated as Verified Doctor!")
                    st.rerun()
                else:
                    st.error(message)

        with dcol3:
            st.markdown("#### ⚙️ Demo Administrator")
            st.write(f"**Email**: `{demo_admin_email}`  \n**Data**: Doctor Approval Queue, User Control, Audit Logs")
            if st.button("⚡ Login as Demo Admin", use_container_width=True, type="secondary"):
                success, message, user = login_user(demo_admin_email, demo_admin_password)
                if success and user:
                    set_session(user)
                    st.session_state["page"] = "admin_dashboard"
                    st.success("Authenticated as Administrator!")
                    st.rerun()
                else:
                    st.error(message)

        with dcol4:
            st.markdown("#### ⏳ Pending Doctor")
            st.write(f"**Email**: `{demo_pending_doctor_email}`  \n**Data**: Pending Application, Uploaded Credentials")
            if st.button("⚡ Login as Pending Doctor", use_container_width=True, type="secondary"):
                success, message, user = login_user(demo_pending_doctor_email, demo_pending_doctor_password)
                if success and user:
                    set_session(user)
                    st.session_state["page"] = "doctor_pending"
                    st.success("Authenticated as Pending Doctor!")
                    st.rerun()
                else:
                    st.error(message)

        st.markdown("---")
        st.info("💡 **Pending Doctor Application Demo**: To demonstrate Doctor Document Review and Verification, log in as **Admin**, click **Doctor Verifications**, and inspect `pending.doctor@example.com` (`dr_pending_demo`).")
        _card_end()



def _save_scan_result(
    result: dict,
    uploaded_file,
    patient_id: int | None = None,
    doctor_id: int | None = None,
    target_user_id: int | None = None,
) -> int:
    segmentation = result.get("segmentation") or {}
    user_id = target_user_id if target_user_id is not None else current_user_id()
    return save_prediction(
        user_id=user_id,
        patient_id=patient_id,
        doctor_id=doctor_id,
        image_filename=uploaded_file.name,
        image_data=result["steps"]["file_bytes"],
        prediction=result["class"],
        confidence=result["confidence"],
        class_probs=result["probabilities"],
        severity=result["severity"],
        confidence_label=result["confidence_text"],
        quality_label=result["quality"]["label"],
        focus_region=result["focus_region"],
        ai_summary=result["ai_summary"],
        assistant_text=result["assistant_text"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        segmentation_coverage=segmentation.get("coverage_ratio"),
    )


def _render_result_visuals(result: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    info = CLASS_INFO[result["class"]]
    _metric_tile(col1, result["class"], "Prediction", info["color"])
    _metric_tile(col2, f"{result['confidence'] * 100:.1f}%", "Confidence", "#16b9ff")
    _metric_tile(col3, result["severity"], "Severity", "#fcbf49")
    _metric_tile(col4, result["quality"]["label"], "MRI Quality", "#00d4a5")

    prob_df = pd.DataFrame(
        {"Class": list(result["probabilities"].keys()), "Probability": [value * 100 for value in result["probabilities"].values()]}
    )
    left, right = st.columns(2, gap="large")
    with left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=result["confidence"] * 100,
                number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": info["color"]}},
                title={"text": "Confidence"},
            )
        )
        st.plotly_chart(gauge, use_container_width=True)
    with right:
        fig = px.bar(prob_df, x="Probability", y="Class", orientation="h", color="Class", text="Probability")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


def page_forgot_password() -> None:
    aside_html = _render_auth_frame(
        "Intelligence, with care",
        "A secure way back to your workspace.",
        "Verify your email address and choose a new password to get back to your care journey.",
    )
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown(aside_html, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="auth-content"><h2>Reset your password</h2><p class="auth-content-intro">We’ll send a one-time recovery code to your email.</p></div>', unsafe_allow_html=True)

        with st.form("password_reset_request_form"):
            email = st.text_input(
                "Email address",
                value=st.session_state.get("password_reset_email", ""),
                placeholder="name@example.com",
            )
            request_submitted = st.form_submit_button("Send recovery code")

        if request_submitted:
            requested, message = request_password_reset(email)
            if requested:
                st.session_state["password_reset_email"] = email.strip().lower()
                st.session_state["password_reset_requested"] = True
                st.info(message)
                if os.getenv("APP_ENV", "development").lower() != "production" and not (
                    os.getenv("SMTP_USERNAME") and os.getenv("SMTP_PASSWORD")
                ):
                    st.caption("Development mode: check the app server output for the demo recovery code.")
            else:
                st.error(message)

        if st.session_state.get("password_reset_requested"):
            with st.form("password_reset_complete_form"):
                st.text_input(
                    "Recovery code",
                    placeholder="Enter the 6-digit code",
                    max_chars=6,
                    key="password_reset_code",
                )
                st.text_input(
                    "New password",
                    type="password",
                    placeholder="At least 8 characters, a number and a symbol",
                    key="password_reset_new_password",
                )
                st.text_input(
                    "Confirm new password",
                    type="password",
                    placeholder="Re-enter your new password",
                    key="password_reset_confirm_password",
                )
                reset_submitted = st.form_submit_button("Update password")

            if reset_submitted:
                reset, message = complete_password_reset(
                    st.session_state["password_reset_email"],
                    st.session_state.get("password_reset_code", ""),
                    st.session_state.get("password_reset_new_password", ""),
                    st.session_state.get("password_reset_confirm_password", ""),
                )
                if reset:
                    for key in (
                        "password_reset_email",
                        "password_reset_requested",
                        "password_reset_code",
                        "password_reset_new_password",
                        "password_reset_confirm_password",
                    ):
                        st.session_state.pop(key, None)
                    st.success(message)
                    st.markdown('<div class="auth-inline"><a href="?page=login">Return to sign in</a></div>', unsafe_allow_html=True)
                else:
                    st.error(message)

        st.markdown('<div class="auth-inline"><a href="?page=login">Back to sign in</a></div>', unsafe_allow_html=True)


def page_upload() -> None:
    if not is_logged_in():
        st.warning("Please log in to upload an MRI scan.")
        return

    model_available = is_model_available()
    if not model_available:
        st.warning(
            "MRI predictions are disabled because the trained model file is not available. "
            "Install models/exports/best_model.keras to enable scan analysis."
        )

    patient_layout = is_patient()
    if patient_layout:
        _patient_workspace_header(
            "Upload MRI",
            "Upload your MRI. See it clearly.",
            "Your original image stays linked to your scan history and available reports.",
        )
    else:
        _hero("Upload & Scan", "Run MRI preprocessing, classification, explainability, and segmentation")
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        with st.container(border=patient_layout):
            st.subheader("Choose an MRI image")
            st.caption("JPG or PNG · up to 10 MB")
            uploaded_file = st.file_uploader(
                "Drop MRI files here or browse",
                type=["jpg", "jpeg", "png"],
                help="Upload a JPG or PNG image. DICOM and NIfTI files are not supported yet.",
            )
            st.caption("Remove identifying file metadata where appropriate before uploading.")
            with st.expander("Analysis options"):
                show_preprocessing = st.checkbox("Show preprocessing visuals", value=True)
                enable_segmentation = st.checkbox("Enable region highlight (heuristic)", value=True)
                enable_gradcam = st.checkbox("Enable Grad-CAM", value=True)
            patient_name = ""
            patient_age = 0
            patient_gender = "Prefer not to say"
            notes = ""
            if not patient_layout:
                st.markdown("Optional patient details")
                patient_name = st.text_input("Patient name")
                patient_age = st.number_input("Patient age", min_value=0, max_value=120, value=35)
                patient_gender = st.selectbox("Patient gender", ["Prefer not to say", "Female", "Male", "Other"])
                notes = st.text_area("Clinical notes")
            run_btn = st.button(
                "Analyze MRI",
                type="primary",
                use_container_width=True,
                disabled=uploaded_file is None or not model_available,
            )

    with right:
        with st.container(border=patient_layout):
            st.subheader("Your scan, step by step")
            for title, detail in (
                ("1 · MRI uploaded", "Image format and size are checked first."),
                ("2 · AI analysis", "The model output is shown as decision support only."),
                ("3 · Review together", "Share a report with a doctor through an accepted care relationship."),
                ("4 · Keep your history", "Your saved scan and reports stay in your account."),
            ):
                st.markdown(f"**{title}**")
                st.caption(detail)
            st.markdown(
                '<div class="patient-warning">AI-generated results are not a diagnosis. '
                'Discuss MRI findings with a qualified medical professional.</div>',
                unsafe_allow_html=True,
            )

    if not uploaded_file:
        return

    valid, message = validate_image(uploaded_file)
    if not valid:
        st.error(message)
        return

    img_array = load_image_from_upload(uploaded_file)
    is_mri, ood_message = is_probable_brain_mri(img_array)
    if not is_mri:
        st.error(f"❌ Out-of-Distribution Error: {ood_message}")
        st.info("Please upload a standard grayscale Brain MRI scan for diagnostic classification.")
        return

    st.image(uploaded_file, caption="MRI preview", use_column_width=True)
    if not run_btn:
        return

    patient_id = None
    if not patient_layout and patient_name.strip():
        patient_id = create_patient_record(current_user_id(), patient_name.strip(), int(patient_age), patient_gender, notes or None)

    progress = st.progress(0, text="Starting NeuroScan pipeline")
    progress.progress(15, text="Preprocessing MRI")
    result = analyze_uploaded_scan(uploaded_file, enable_gradcam=enable_gradcam, enable_segmentation=enable_segmentation)
    progress.progress(70, text="Saving prediction and preparing report")
    prediction_id = _save_scan_result(result, uploaded_file, patient_id)
    progress.progress(100, text="Analysis complete")
    st.session_state["last_scan_result"] = result

    _card_start()
    st.subheader("Prediction Summary")
    st.success(f"Scan #{prediction_id:06d} classified as {result['class']}")

    if result.get("status") == "low_confidence" or result.get("low_confidence_warning"):
        st.warning(
            result.get(
                "low_confidence_warning",
                f"⚠️ **Low Diagnostic Confidence Warning**: Calibrated confidence ({result['confidence']*100:.1f}%) "
                "is below the clinical threshold of 70%. **Consultation with a medical specialist / neuro-oncologist is strongly advised.**",
            )
        )

    st.write(result["ai_summary"])
    st.caption(result["assistant_text"])
    _card_end()

    _render_result_visuals(result)

    steps = result["steps"]
    if show_preprocessing:
        st.subheader("Preprocessing Review")
        fig = plot_preprocessing_steps(steps)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        hist_fig = plot_pixel_histogram(steps["normalized"])
        st.pyplot(hist_fig, use_container_width=True)
        plt.close(hist_fig)

    image_count = 2 + int(result.get("gradcam") is not None) + int(bool(result.get("segmentation")))
    image_cols = st.columns(image_count)
    image_cols[0].image(steps["original"], caption="Original")
    image_cols[1].image(steps["resized"], caption="Preprocessed")
    idx = 2
    if result.get("gradcam") is not None:
        image_cols[idx].image(result["gradcam"], caption="Grad-CAM")
        idx += 1
    if result.get("segmentation"):
        method_label = result["segmentation"].get("method", "Region highlight (heuristic)")
        image_cols[idx].image(result["segmentation"]["overlay"], caption=method_label)

    patient_notes = get_clinical_notes_for_prediction(prediction_id)
    patient_recommendations = [
        note["recommendation"]
        for note in patient_notes
        if note.get("recommendation")
    ] or ["No clinician recommendation has been recorded for this scan."]
    report_ai = _report_ai_fields(
        result["class"],
        float(result["confidence"]),
        result["probabilities"],
        result.get("ai_summary"),
    )
    pdf_bytes = generate_pdf_report(
        username=current_username(),
        scan_date=datetime.now().strftime("%d %B %Y %H:%M:%S"),
        original_img=steps["original"],
        preprocessed_img=steps["resized"],
        gradcam_img=result.get("gradcam"),
        segmentation_img=(result.get("segmentation") or {}).get("overlay"),
        prediction=result["class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        class_info={**CLASS_INFO[result["class"]], "severity": result["severity"]},
        recommendations=patient_recommendations,
        clinical_notes=patient_notes,
        scan_id=prediction_id,
        ai_summary=report_ai["summary"],
        ai_sources=report_ai["sources"],
        research_status=report_ai["status"],
        model_name=result["model_name"],
        quality_label=result["quality"]["label"],
    )
    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name=f"NeuroScan_Report_{prediction_id:06d}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def page_dashboard() -> None:
    if not is_logged_in():
        st.warning("Please log in to view the dashboard.")
        return

    stats = get_dashboard_stats(current_user_id())
    if is_patient():
        user = get_user_by_id(current_user_id())
        display_name = _patient_display_name(user)
        records = get_user_predictions(current_user_id(), limit=100)
        doctors = get_assigned_doctors_for_patient(current_user_id())
        available_reports = sum(_patient_report_available(record) for record in records)
        notes_by_scan = {
            record["id"]: get_clinical_notes_for_prediction(record["id"])
            for record in records
        }
        reviewed_scans = sum(bool(notes) for notes in notes_by_scan.values())
        awaiting_review = max(0, len(records) - reviewed_scans)
        _patient_workspace_header(
            "Dashboard",
            f"Good morning, {display_name}",
            "Your scans, care team and next steps — all in one place.",
        )
        if user and user.get("username") == "demo_patient":
            st.markdown(
                '<div class="patient-warning"><strong>Illustrative demo account.</strong> '
                'Any sample scan record is mock data, not a real medical record or diagnosis.</div>',
                unsafe_allow_html=True,
            )
        metric_columns = st.columns(4)
        metric_values = [
            (str(stats["total"]), "MRI scans"),
            (str(available_reports), "Reports available"),
            (f"{len(doctors):02d}", "Your care team"),
            (f"{awaiting_review:02d}", "Awaiting clinician review"),
        ]
        for column, (value, label) in zip(metric_columns, metric_values):
            column.markdown(
                f'<div class="patient-metric"><div class="patient-metric-value">{value}</div>'
                f'<div class="patient-metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )
        action_col, history_col, report_col = st.columns([1, 1, 1])
        if action_col.button("＋  Upload MRI", type="primary", use_container_width=True):
            st.session_state["page"] = "upload"
            st.rerun()
        if history_col.button("View scan history", use_container_width=True):
            st.session_state["page"] = "history"
            st.rerun()
        if report_col.button("Open reports", use_container_width=True):
            st.session_state["page"] = "patient_reports"
            st.rerun()

        latest = records[0] if records else None
        left, right = st.columns([2, 1.05], gap="medium")
        with left:
            with st.container(border=True):
                st.markdown("#### Your latest scan")
                if latest:
                    st.caption(
                        f"NS-{latest['id']:06d} · {latest.get('image_filename') or 'MRI scan'} · "
                        f"{latest.get('created_at') or 'Date unavailable'}"
                    )
                    image_bytes = get_prediction_by_id(latest["id"])
                    image_bytes = image_bytes.get("image_data") if image_bytes else None
                    if image_bytes:
                        try:
                            scan_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        except (UnidentifiedImageError, OSError):
                            st.warning("The saved MRI image is unavailable or could not be opened.")
                        else:
                            st.image(scan_image, caption="Original uploaded MRI", use_column_width=True)
                    result_col, status_col = st.columns([1.4, 1])
                    result_col.markdown(
                        f"**{latest.get('prediction') or 'Analysis saved'}** · "
                        f"{float(latest.get('confidence') or 0) * 100:.1f}% model confidence"
                    )
                    result_col.caption(
                        f"MRI quality: {latest.get('quality_label') or 'Not recorded'} · "
                        f"{latest.get('severity') or 'Review with your doctor'}"
                    )
                    status_col.caption(
                        "Clinician review recorded"
                        if notes_by_scan.get(latest["id"])
                        else "No clinician note recorded"
                    )
                    if st.button("View scan details", key="dashboard_latest_scan"):
                        st.session_state["selected_patient_scan_id"] = latest["id"]
                        st.session_state["page"] = "history"
                        st.rerun()
                else:
                    st.info("Your scan history is empty. Upload an MRI to start a record.")
                    if st.button("Upload your first MRI", type="primary", key="dashboard_first_upload"):
                        st.session_state["page"] = "upload"
                        st.rerun()
                st.markdown(
                    '<div class="patient-warning">AI predictions require clinical review. '
                    'Model confidence is not a diagnosis.</div>',
                    unsafe_allow_html=True,
                )
        with right:
            with st.container(border=True):
                st.markdown("#### Your care team")
                if doctors:
                    for doctor in doctors[:3]:
                        st.markdown(f"**Dr. {html.escape(_doctor_display_name(doctor))}**")
                        st.caption(
                            f"{doctor.get('specialization') or 'Verified doctor'}"
                            + (f" · {doctor['hospital']}" if doctor.get("hospital") else "")
                        )
                else:
                    st.info("No active doctor relationships yet.")
                if st.button("View My Doctors", use_container_width=True):
                    st.session_state["page"] = "patient_doctors"
                    st.rerun()
            with st.container(border=True):
                st.markdown("#### Mastishka AI")
                st.caption("Understand MRI model output and prepare questions for your doctor.")
                if st.button("Ask about my scan ↗", use_container_width=True):
                    st.session_state["page"] = "mastishka"
                    st.rerun()
        lower_left, lower_right = st.columns([1.6, 1], gap="medium")
        with lower_left:
            with st.container(border=True):
                st.markdown("#### Scan history & reports")
                if records:
                    history_rows = [
                        {
                            "Scan": f"NS-{record['id']:06d}",
                            "Date": record.get("created_at") or "—",
                            "AI output": record.get("prediction") or "—",
                            "Model confidence": f"{float(record.get('confidence') or 0) * 100:.1f}%",
                            "Clinical review": (
                                "Reviewed by clinician"
                                if notes_by_scan.get(record["id"])
                                else "Awaiting review"
                            ),
                        }
                        for record in records[:5]
                    ]
                    st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
                    if st.button("View all history", key="dashboard_all_history"):
                        st.session_state["page"] = "history"
                        st.rerun()
                else:
                    st.info("Your saved scans and reports will appear here.")
        with lower_right:
            with st.container(border=True):
                st.markdown("#### From your care team")
                clinical_notes = [
                    (record, note)
                    for record in records
                    for note in notes_by_scan.get(record["id"], [])
                ]
                if clinical_notes:
                    for record, note in clinical_notes[:2]:
                        st.markdown(
                            f"**Dr. {html.escape(_doctor_display_name({'username': note['doctor_name']}))}** "
                            f"· NS-{record['id']:06d}"
                        )
                        if note.get("recommendation"):
                            st.caption(f"Recommendation: {note['recommendation']}")
                        st.write(note.get("note_content") or "")
                else:
                    st.info("Signed clinician notes will appear here when available.")
                if st.button("Open consultation", use_container_width=True):
                    st.session_state["page"] = "consultation"
                    st.rerun()
        return

    _hero("Dashboard", f"Welcome back, {current_username()}")
    c1, c2, c3, c4 = st.columns(4)
    _metric_tile(c1, str(stats["total"]), "Total Scans")
    _metric_tile(c2, str(sum(count for cls, count in stats["by_class"].items() if cls != "Normal")), "Tumour Cases", "#ff6b6b")
    _metric_tile(c3, str(stats["by_class"].get("Normal", 0)), "Normal Cases", "#00d4a5")
    _metric_tile(c4, str(len(stats.get("models_used", {}))), "Models Used", "#fcbf49")

    left, right = st.columns(2, gap="large")
    with left:
        _card_start()
        st.subheader("Prediction Distribution")
        if stats["by_class"]:
            pie = px.pie(values=list(stats["by_class"].values()), names=list(stats["by_class"].keys()), hole=0.45)
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("No predictions yet.")
        _card_end()

    with right:
        _card_start()
        st.subheader("MRI Quality Breakdown")
        if stats["quality_breakdown"]:
            quality_df = pd.DataFrame({"Quality": list(stats["quality_breakdown"].keys()), "Count": list(stats["quality_breakdown"].values())})
            quality_fig = px.bar(quality_df, x="Quality", y="Count", color="Quality")
            st.plotly_chart(quality_fig, use_container_width=True)
        else:
            st.info("No quality analytics yet.")
        _card_end()

    if stats["recent"]:
        st.subheader("Recent Scans")
        st.dataframe(pd.DataFrame(stats["recent"]), use_container_width=True, hide_index=True)


def page_history() -> None:
    if not is_logged_in():
        st.warning("Please log in to view scan history.")
        return

    patient_layout = is_patient()
    if patient_layout:
        _patient_workspace_header(
            "Scan History",
            "Your imaging timeline",
            "Explore your scans, review status and available clinical context.",
        )
    else:
        _hero("Scan History", "All stored MRI analysis records")
    records = get_user_predictions(current_user_id(), limit=200)
    if not records:
        with st.container(border=patient_layout):
            st.info("No previous scans found. Your saved MRI scans will appear here.")
            if patient_layout and st.button("Upload MRI", type="primary"):
                st.session_state["page"] = "upload"
                st.rerun()
        return
    if patient_layout:
        latest_notes = {
            record["id"]: get_clinical_notes_for_prediction(record["id"])
            for record in records
        }
        scan_query = st.text_input(
            "Search scans",
            placeholder="Search by scan ID, date, file name or AI output...",
        )
        filtered_records = [
            record for record in records
            if scan_query.strip().lower() in (
                f"NS-{record['id']:06d} {record.get('created_at', '')} "
                f"{record.get('image_filename', '')} {record.get('prediction', '')}"
            ).lower()
        ]
        with st.container(border=True):
            st.markdown("#### Scan records")
            if not filtered_records:
                st.info("No scans match your search.")
                return
            history_rows = [
                {
                    "Scan": f"NS-{record['id']:06d}",
                    "Date": record.get("created_at") or "—",
                    "File": record.get("image_filename") or "—",
                    "AI output": record.get("prediction") or "—",
                    "Confidence": f"{float(record.get('confidence') or 0) * 100:.1f}%",
                    "Clinical status": (
                        "Reviewed by doctor"
                        if latest_notes.get(record["id"])
                        else "Awaiting review"
                    ),
                }
                for record in filtered_records
            ]
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
        selected_scan_id = st.session_state.get("selected_patient_scan_id")
        selected_record = next(
            (record for record in filtered_records if record["id"] == selected_scan_id),
            filtered_records[0],
        )
        st.session_state["selected_patient_scan_id"] = selected_record["id"]
        with st.container(border=True):
            st.markdown(f"#### Selected scan · NS-{selected_record['id']:06d}")
            st.caption(
                f"{selected_record.get('created_at') or 'Date unavailable'} · "
                f"{selected_record.get('image_filename') or 'MRI scan'}"
            )
            details_col, activity_col = st.columns([1.5, 1])
            with details_col:
                record = get_prediction_by_id(selected_record["id"])
                image_bytes = record.get("image_data") if record else None
                if image_bytes:
                    try:
                        scan_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    except (UnidentifiedImageError, OSError):
                        st.warning("The saved MRI image is unavailable or could not be opened.")
                    else:
                        st.image(scan_image, caption="Original uploaded MRI", use_column_width=True)
                st.markdown(
                    f"**{selected_record.get('prediction') or 'Analysis saved'}** · "
                    f"{float(selected_record.get('confidence') or 0) * 100:.1f}% model confidence"
                )
                st.caption(
                    f"MRI quality: {selected_record.get('quality_label') or 'Not recorded'} · "
                    f"Model: {selected_record.get('model_name') or 'Not recorded'}"
                )
                if record and record.get("ai_summary"):
                    st.write(record["ai_summary"])
            with activity_col:
                st.markdown("##### Clinical review")
                scan_notes = latest_notes.get(selected_record["id"], [])
                if scan_notes:
                    for note in scan_notes:
                        st.markdown(
                            f"**Dr. {html.escape(_doctor_display_name({'username': note['doctor_name']}))}**"
                        )
                        st.caption(note.get("created_at") or "Date unavailable")
                        if note.get("recommendation"):
                            st.markdown(f"**Recommendation:** {note['recommendation']}")
                        st.write(note.get("note_content") or "")
                else:
                    st.info("No signed clinician note is attached to this scan yet.")
                if st.button("Open related report", key=f"history_report_{selected_record['id']}"):
                    st.session_state["selected_patient_scan_id"] = selected_record["id"]
                    st.session_state["page"] = "patient_reports"
                    st.rerun()
        return
    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _patient_report_bytes(record: dict, user: dict) -> bytes:
    image_data = record.get("image_data")
    if not image_data:
        raise ValueError("This scan does not contain an MRI image for report generation.")
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            "The saved MRI image is unavailable, so a PDF report cannot be generated."
        ) from exc
    steps = preprocess_array(np.asarray(image))
    prediction = str(record.get("prediction") or "")
    if prediction not in CLASS_INFO:
        raise ValueError("The saved scan has an unknown model result; a report cannot be generated.")
    notes = get_clinical_notes_for_prediction(record["id"])
    probabilities = {
        "Glioma": float(record.get("glioma_prob") or 0),
        "Meningioma": float(record.get("meningioma_prob") or 0),
        "Pituitary": float(record.get("pituitary_prob") or 0),
        "Normal": float(record.get("normal_prob") or 0),
    }
    report_ai = _report_ai_fields(
        prediction,
        float(record.get("confidence") or 0),
        probabilities,
        record.get("ai_summary"),
    )
    return generate_pdf_report(
        username=_patient_display_name(user),
        scan_date=str(record.get("created_at") or "Date unavailable"),
        original_img=steps["original"],
        preprocessed_img=steps["resized"],
        gradcam_img=None,
        segmentation_img=None,
        prediction=prediction,
        confidence=float(record.get("confidence") or 0),
        probabilities=probabilities,
        class_info={**CLASS_INFO[prediction], "severity": record.get("severity") or "Not assessed"},
        scan_id=int(record["id"]),
        ai_summary=report_ai["summary"],
        ai_sources=report_ai["sources"],
        research_status=report_ai["status"],
        recommendations=[
            "Discuss this AI-generated result with a qualified clinician and correlate it with the full MRI series and clinical context."
        ],
        clinical_notes=notes,
        model_name=record.get("model_name"),
        quality_label=record.get("quality_label"),
    )


def _patient_report_available(record: dict) -> bool:
    if record.get("prediction") not in CLASS_INFO:
        return False
    details = get_prediction_by_id(record["id"])
    image_data = details.get("image_data") if details else None
    if not image_data:
        return False
    try:
        Image.open(io.BytesIO(image_data)).verify()
    except (UnidentifiedImageError, OSError):
        return False
    return True


def page_patient_reports() -> None:
    if not is_patient():
        st.error("Access restricted: patient reports are available to signed-in patients.")
        return
    user_id = current_user_id()
    user = get_user_by_id(user_id) if user_id else None
    if not user:
        st.error("Patient account is unavailable. Please sign in again.")
        return

    _patient_workspace_header(
        "Reports",
        "Reports, with the full context",
        "Your AI scan reports and signed clinician notes are clearly separated.",
    )
    records = get_user_predictions(user_id, limit=200)
    if user.get("username") == "demo_patient":
        st.markdown(
            '<div class="patient-warning"><strong>Illustrative demo account.</strong> '
            'Sample scan records are mock data and do not represent a real diagnosis.</div>',
            unsafe_allow_html=True,
        )
    if not records:
        with st.container(border=True):
            st.info("No reports are available yet. Upload an MRI to create your first AI report.")
            if st.button("Upload MRI", type="primary"):
                st.session_state["page"] = "upload"
                st.rerun()
        return

    signed_notes = {
        record["id"]: get_clinical_notes_for_prediction(record["id"])
        for record in records
    }
    summary = st.columns(3)
    summary[0].metric("Scans", len(records))
    summary[1].metric(
        "Clinician reviewed",
        sum(bool(notes) for notes in signed_notes.values()),
    )
    summary[2].metric(
        "Awaiting review",
        sum(not notes for notes in signed_notes.values()),
    )
    selected_id = st.session_state.get("selected_patient_scan_id")
    selected_index = next(
        (i for i, record in enumerate(records) if record["id"] == selected_id),
        0,
    )
    selected_record = st.selectbox(
        "Choose a scan report",
        records,
        index=selected_index,
        format_func=lambda record: (
            f"NS-{record['id']:06d} · {record.get('created_at') or 'Date unavailable'} · "
            f"{record.get('prediction') or 'Analysis'}"
        ),
        key="patient_report_select",
    )
    st.session_state["selected_patient_scan_id"] = selected_record["id"]

    report_list, report_detail = st.columns([0.85, 2.15], gap="medium")
    with report_list:
        with st.container(border=True):
            st.markdown("#### Your reports")
            for record in records[:10]:
                st.markdown(f"**NS-{record['id']:06d} · Brain MRI**")
                st.caption(record.get("created_at") or "Date unavailable")
                st.caption(
                    "Signed clinician note available"
                    if signed_notes.get(record["id"])
                    else "AI draft · awaiting clinical review"
                )
                st.markdown("---")
            st.caption("AI-generated reports are not signed clinical interpretations.")
    with report_detail:
        with st.container(border=True):
            st.markdown(f"#### Report preview · NS-{selected_record['id']:06d}")
            scan_data = get_prediction_by_id(selected_record["id"])
            if not scan_data:
                st.error("This scan record could not be loaded.")
                return
            notes = signed_notes.get(selected_record["id"], [])
            st.caption(
                f"{scan_data.get('created_at') or 'Date unavailable'} · "
                f"{scan_data.get('image_filename') or 'MRI scan'}"
            )
            st.markdown(
                f"**{scan_data.get('prediction') or 'Analysis saved'}** · "
                f"{float(scan_data.get('confidence') or 0) * 100:.1f}% model confidence"
            )
            image_data = scan_data.get("image_data")
            if image_data:
                try:
                    preview = Image.open(io.BytesIO(image_data)).convert("RGB")
                except (UnidentifiedImageError, OSError):
                    st.warning("The saved MRI image is not available for a report preview.")
                else:
                    st.image(preview, caption="Original MRI · stored with this scan", use_column_width=True)
            st.markdown("##### Imaging and model limitations")
            st.caption(
                "AI output reflects a model estimate. It is not a diagnosis and must be reviewed "
                "alongside the full imaging series and clinical context."
            )
            if notes:
                st.markdown("##### Latest signed clinician note")
                for note in notes:
                    st.markdown(
                        f"**Dr. {html.escape(_doctor_display_name({'username': note['doctor_name']}))}**"
                    )
                    st.caption(note.get("created_at") or "Date unavailable")
                    if note.get("recommendation"):
                        st.markdown(f"**Recommendation:** {note['recommendation']}")
                    st.write(note.get("note_content") or "")
            else:
                st.info("No signed clinician interpretation is attached to this report.")
            try:
                report_pdf = _patient_report_bytes(scan_data, user)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                st.warning(f"PDF report unavailable: {exc}")
            else:
                st.download_button(
                    "Download PDF report",
                    data=report_pdf,
                    file_name=f"NeuroScan_Report_{scan_data['id']:06d}.pdf",
                    mime="application/pdf",
                    type="primary",
                )


def page_patient_doctors() -> None:
    if not is_patient():
        st.error("Access restricted: the care-team page is for signed-in patients.")
        return
    patient_id = current_user_id()
    _patient_workspace_header(
        "My Doctors",
        "Your care team",
        "Know who can review your scans and manage your patient-authorized relationships.",
    )
    assigned_doctors = get_assigned_doctors_for_patient(patient_id)
    with st.container(border=True):
        st.markdown("#### Active care relationships")
        st.markdown(
            '<div class="patient-callout">Only doctors you connect with can access your scans. '
            'You can review each active relationship here.</div>',
            unsafe_allow_html=True,
        )
        if assigned_doctors:
            doctor_columns = st.columns(min(2, len(assigned_doctors)))
            for index, doctor in enumerate(assigned_doctors):
                with doctor_columns[index % len(doctor_columns)]:
                    with st.container(border=True):
                        st.markdown(f"### Dr. {html.escape(_doctor_display_name(doctor))}")
                        st.caption(
                            f"{doctor.get('specialization') or 'Verified specialist'}"
                            + (f" · {doctor['hospital']}" if doctor.get("hospital") else "")
                        )
                        st.caption(f"Connected {doctor.get('assigned_at') or 'date unavailable'}")
                        st.markdown(f"**Contact**  \n{html.escape(doctor.get('email') or 'Not provided')}")
                        st.caption("Relationship active · patient consent recorded")
                        if st.button(
                            "Schedule consultation",
                            key=f"patient_doctor_schedule_{doctor['id']}",
                            use_container_width=True,
                        ):
                            st.session_state["consultation_doctor_id"] = doctor["id"]
                            st.session_state["page"] = "consultation"
                            st.rerun()
        else:
            st.info("You do not have an active doctor relationship yet.")

    verified_doctors = {
        doctor["id"]: doctor
        for status in ("active", "approved")
        for doctor in get_all_doctors(status=status)
    }
    available_doctors = [
        doctor for doctor in verified_doctors.values()
        if not is_patient_assigned_to_doctor(patient_id, doctor["id"])
    ]
    with st.container(border=True):
        st.markdown("#### Find a verified doctor")
        if not available_doctors:
            st.info("No additional verified doctors are available to connect right now.")
            return
        doctor_by_label = {
            f"Dr. {_doctor_display_name(doctor)} · {doctor.get('specialization') or 'Specialist'}"
            + (f" · {doctor['hospital']}" if doctor.get("hospital") else ""): doctor
            for doctor in available_doctors
        }
        selected_label = st.selectbox(
            "Choose a doctor",
            list(doctor_by_label),
            key="patient_doctor_directory_select",
        )
        st.markdown(
            '<div class="patient-warning">Connecting grants this verified doctor access to your '
            'saved MRI scans and reports so they can provide clinical review.</div>',
            unsafe_allow_html=True,
        )
        consent = st.checkbox(
            "I want to connect with this doctor and share my scan records for clinical review.",
            key="patient_doctor_consent",
        )
        if st.button("Connect with selected doctor", type="primary", disabled=not consent):
            selected_doctor = doctor_by_label[selected_label]
            if assign_doctor_to_patient(patient_id, selected_doctor["id"]):
                st.success("Care relationship established. The doctor can now review your shared records.")
                st.rerun()
            else:
                st.error("The care relationship could not be established. Please retry.")


def _render_eval_metrics_summary(metrics: dict) -> None:
    """Render overall metrics, tumour recall highlights, and per-class performance table."""
    c1, c2, c3, c4 = st.columns(4)
    _metric_tile(c1, "VGG16 Transfer Model", "Evaluated Model")
    _metric_tile(c2, f"{metrics['accuracy'] * 100:.1f}%", "Overall Accuracy", "#00d4a5")
    _metric_tile(c3, f"{metrics['macro_avg']['f1_score'] * 100:.1f}%", "Macro F1 Score", "#16b9ff")
    _metric_tile(c4, str(metrics.get("total_samples", 0)), "Test Samples", "#fcbf49")

    with st.container(border=True):
        st.subheader("Tumour class recall highlights")
        st.caption(
            "Recall (sensitivity) measures the share of cases in the evaluation set "
            "correctly identified by the model; it is not a clinical outcome guarantee."
        )

        r_col1, r_col2, r_col3 = st.columns(3)
        recalls = metrics.get("tumour_recalls", {})
        per_cls = metrics.get("per_class", {})

        glioma_rec = recalls.get("Glioma", per_cls.get("Glioma", {}).get("recall", 0.0))
        meningioma_rec = recalls.get("Meningioma", per_cls.get("Meningioma", {}).get("recall", 0.0))
        pituitary_rec = recalls.get("Pituitary", per_cls.get("Pituitary", {}).get("recall", 0.0))

        _metric_tile(r_col1, f"{glioma_rec * 100:.1f}%", "Glioma Recall", "#ff6b6b")
        _metric_tile(r_col2, f"{meningioma_rec * 100:.1f}%", "Meningioma Recall", "#ffd43b")
        _metric_tile(r_col3, f"{pituitary_rec * 100:.1f}%", "Pituitary Recall", "#74c0fc")

    with st.container(border=True):
        st.subheader("Detailed classification performance")
        if per_cls:
            df_rows = []
            for cls_name, stat in per_cls.items():
                df_rows.append({
                    "Class": cls_name,
                    "Precision": f"{stat['precision'] * 100:.2f}%",
                    "Recall (Sensitivity)": f"{stat['recall'] * 100:.2f}%",
                    "F1-Score": f"{stat['f1_score'] * 100:.2f}%",
                    "Support (Samples)": stat["support"],
                })
            if "macro_avg" in metrics:
                df_rows.append({
                    "Class": "Macro Average",
                    "Precision": f"{metrics['macro_avg']['precision'] * 100:.2f}%",
                    "Recall (Sensitivity)": f"{metrics['macro_avg']['recall'] * 100:.2f}%",
                    "F1-Score": f"{metrics['macro_avg']['f1_score'] * 100:.2f}%",
                    "Support (Samples)": metrics.get("total_samples", "-"),
                })
            if "weighted_avg" in metrics:
                df_rows.append({
                    "Class": "Weighted Average",
                    "Precision": f"{metrics['weighted_avg']['precision'] * 100:.2f}%",
                    "Recall (Sensitivity)": f"{metrics['weighted_avg']['recall'] * 100:.2f}%",
                    "F1-Score": f"{metrics['weighted_avg']['f1_score'] * 100:.2f}%",
                    "Support (Samples)": metrics.get("total_samples", "-"),
                })
            st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)


def _render_eval_plots() -> None:
    """Display confusion matrix and One-vs-Rest ROC curve plots."""
    cm_path = Path("logs/confusion_matrix.png")
    if not cm_path.exists():
        cm_path = Path("data/artifacts/plots/confusion_matrix.png")

    roc_path = Path("logs/roc_curves.png")
    if not roc_path.exists():
        roc_path = Path("data/artifacts/plots/roc_curves.png")

    p_col1, p_col2 = st.columns(2, gap="large")
    with p_col1:
        with st.container(border=True):
            st.subheader("Confusion matrix")
            if cm_path.exists():
                st.image(str(cm_path), caption="Testing split confusion matrix", use_column_width=True)
            else:
                st.info("Confusion matrix plot not found.")

    with p_col2:
        with st.container(border=True):
            st.subheader("One-vs-rest ROC curves")
            if roc_path.exists():
                st.image(str(roc_path), caption="Multi-class one-vs-rest ROC curves & AUC", use_column_width=True)
            else:
                st.info("ROC curves plot not found.")


def page_training() -> None:
    """Render Training Hub page with evaluation analytics, tumour recall highlights, and plots."""
    if not is_logged_in():
        st.warning("Please log in to view training analytics.")
        return

    if is_doctor():
        _doctor_workspace_header(
            "Training Hub",
            "Explore the model evaluation",
            "Review saved metrics and plots from the available evaluation run.",
        )
        st.markdown(
            '<div class="doctor-callout">These are saved model-evaluation results, not a live training control or a guarantee of clinical accuracy. '
            'They describe the evaluation dataset only.</div>',
            unsafe_allow_html=True,
        )
    else:
        _hero("Training Hub", "Model evaluation, confusion matrix, ROC curves, and tumour recall metrics")

    eval_json_path = Path("logs/evaluation_metrics.json")
    if not eval_json_path.exists():
        eval_json_path = Path("data/artifacts/plots/evaluation_metrics.json")

    summary_path = Path("logs/training_summary.json")

    if not eval_json_path.exists() and not summary_path.exists():
        with st.container(border=True):
            st.info("No evaluation outputs found yet. Run `python model/evaluate.py` to evaluate the model on the Testing split and generate confusion matrix and ROC curves.")
        return

    if eval_json_path.exists():
        eval_metrics = json.loads(eval_json_path.read_text(encoding="utf-8"))
        _render_eval_metrics_summary(eval_metrics)
        _render_eval_plots()

    if summary_path.exists():
        st.markdown("---")
        with st.container(border=True):
            st.subheader("Model architecture comparison")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if "leaderboard" in summary:
                leaderboard = pd.DataFrame(summary["leaderboard"])
                st.dataframe(leaderboard, use_container_width=True, hide_index=True)
                comparison = px.bar(
                    leaderboard.melt(id_vars="model_name", value_vars=["accuracy", "precision", "recall", "f1_score"]),
                    x="model_name",
                    y="value",
                    color="variable",
                    barmode="group",
                )
                st.plotly_chart(comparison, use_container_width=True)


def page_about() -> None:
    _hero("About NeuroScan AI", "Research-grade educational platform for brain MRI analysis")
    _card_start()
    st.markdown(
        """
        NeuroScan AI is a Streamlit-based full-stack brain tumour analysis platform built around:

        - Secure authentication with OTP verification
        - SQLite-backed patient and scan analytics
        - Multi-model deep learning support
        - Explainability via Grad-CAM
        - Segmentation overlays for visual emphasis
        - PDF reporting and dashboard analytics

        Medical disclaimer: this application is intended for educational and research use only and must not be used as a sole diagnostic tool.
        """
    )
    _card_end()


def page_mastishka() -> None:
    scan_context = None
    chat_context_id = f"user:{current_user_id()}"
    selected_patient = None
    selected_scan = None
    doctor_id = current_user_id() if is_doctor() else None
    doctor_layout = doctor_id is not None
    patient_layout = is_patient()
    if is_doctor() and doctor_id is None:
        st.error("Doctor session is unavailable. Please sign in again.")
        return
    if doctor_layout:
        _doctor_workspace_header(
            "Mastishka AI",
            "Clinical context, clearly explained",
            "Explore MRI model output and limitations without substituting for clinical judgment.",
        )
    elif patient_layout:
        _patient_workspace_header(
            "Mastishka AI",
            "Understand the results, together",
            "Ask questions about a selected scan, model output and your next conversation with a doctor.",
        )
    else:
        _hero("🧠 Mastishka Assistant", "Conversational AI Medical Assistant & NeuroScan Knowledge Base")

    if doctor_layout:
        context_column, conversation_column = st.columns([0.85, 2.15], gap="medium")
        with context_column:
            with st.container(border=True):
                st.subheader("Scan context")
                st.caption("Explain a scan without losing its context.")
                patients = get_doctor_assigned_patients_with_details(doctor_id)
                if patients:
                    selected_patient = st.selectbox(
                        "Select patient",
                        patients,
                        format_func=lambda patient: (
                            f"{patient['full_name']} · {patient['patient_code']}"
                        ),
                        key="mastishka_patient_context",
                    )
                    scans = (
                        get_user_predictions(selected_patient["id"], limit=100)
                        if selected_patient
                        else []
                    )
                    if scans:
                        selected_scan = st.selectbox(
                            "Selected scan",
                            scans,
                            format_func=lambda scan: (
                                f"Scan #{scan['id']} · {scan['created_at']} · "
                                f"{scan['prediction']}"
                            ),
                            key="mastishka_scan_context",
                        )
                        scan_record = (
                            get_prediction_by_id(selected_scan["id"])
                            if selected_scan
                            else None
                        )
                        if scan_record:
                            scan_context = {
                                "class": scan_record["prediction"],
                                "confidence": float(scan_record["confidence"]),
                                "confidence_label": scan_record.get("confidence_label"),
                                "severity": scan_record.get("severity"),
                                "quality": {"label": scan_record.get("quality_label")},
                                "focus_region": scan_record.get("focus_region"),
                                "ai_summary": scan_record.get("ai_summary"),
                            }
                            st.markdown(
                                f"**{html.escape(selected_patient['full_name'])}**  \n"
                                f"{html.escape(selected_patient['patient_code'])}"
                            )
                            if scan_record.get("image_data"):
                                try:
                                    context_image = Image.open(
                                        io.BytesIO(scan_record["image_data"])
                                    ).convert("RGB")
                                except (UnidentifiedImageError, OSError):
                                    st.warning("The stored scan image could not be opened.")
                                else:
                                    st.image(
                                        context_image,
                                        caption="Selected MRI · original image",
                                        use_column_width=True,
                                    )
                            st.caption(
                                f"AI output · {scan_record['prediction']} · "
                                f"{float(scan_record['confidence']) * 100:.1f}% model confidence"
                            )
                            st.caption(
                                "Confidence is model output, not a diagnosis."
                            )
                            chat_context_id = (
                                f"doctor:{doctor_id}:patient:{selected_patient['id']}:"
                                f"scan:{scan_record['id']}"
                            )
                    elif selected_patient:
                        st.info(
                            f"No MRI scans are recorded for {selected_patient['full_name']}."
                        )
                        chat_context_id = (
                            f"doctor:{doctor_id}:patient:{selected_patient['id']}:no-scan"
                        )
                else:
                    st.info(
                        "No accepted patient relationship is available. You can still ask general educational questions."
                    )
                    chat_context_id = f"doctor:{doctor_id}:no-patient"

            with st.container(border=True):
                st.subheader("Current conversation")
                st.caption(
                    "Conversation messages are held for this signed-in session and clinical context."
                )
                if st.button(
                    "＋ New conversation",
                    key="mastishka_new_conversation",
                    use_container_width=True,
                ):
                    st.session_state["mastishka_context_id"] = chat_context_id
                    st.session_state["mastishka_chat"] = []
                    st.rerun()

    elif patient_layout:
        context_column, conversation_column = st.columns([0.85, 2.15], gap="medium")
        with context_column:
            with st.container(border=True):
                st.subheader("Scan context")
                st.caption("Choose a saved scan to keep explanations grounded.")
                patient_scans = get_user_predictions(current_user_id(), limit=100)
                if patient_scans:
                    selected_scan = st.selectbox(
                        "Selected scan",
                        patient_scans,
                        format_func=lambda scan: (
                            f"NS-{scan['id']:06d} · {scan.get('created_at') or 'Date unavailable'}"
                        ),
                        key="mastishka_patient_scan",
                    )
                    scan_record = get_prediction_by_id(selected_scan["id"])
                    if scan_record:
                        scan_context = {
                            "class": scan_record["prediction"],
                            "confidence": float(scan_record["confidence"]),
                            "confidence_label": scan_record.get("confidence_label"),
                            "severity": scan_record.get("severity"),
                            "quality": {"label": scan_record.get("quality_label")},
                            "focus_region": scan_record.get("focus_region"),
                            "ai_summary": scan_record.get("ai_summary"),
                        }
                        image_data = scan_record.get("image_data")
                        if image_data:
                            try:
                                context_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                            except (UnidentifiedImageError, OSError):
                                st.warning("The saved MRI image is unavailable or could not be opened.")
                            else:
                                st.image(
                                    context_image,
                                    caption="Original MRI · selected scan",
                                    use_column_width=True,
                                )
                        st.markdown(
                            f"**{scan_record.get('prediction') or 'Analysis saved'}** · "
                            f"{float(scan_record.get('confidence') or 0) * 100:.1f}% model confidence"
                        )
                        st.caption(
                            f"MRI quality: {scan_record.get('quality_label') or 'Not recorded'}"
                        )
                        st.caption(
                            "Model confidence is not a diagnosis. Review the complete MRI with a clinician."
                        )
                        chat_context_id = (
                            f"patient:{current_user_id()}:scan:{scan_record['id']}"
                        )
                    else:
                        st.error("The selected scan record could not be loaded.")
                else:
                    st.info("Upload an MRI to ask questions about a scan.")
                    chat_context_id = f"patient:{current_user_id()}:no-scan"
            with st.container(border=True):
                st.subheader("Conversation")
                st.caption("This conversation is kept for the current signed-in session.")
                if st.button(
                    "＋ New conversation",
                    key="mastishka_patient_new_conversation",
                    use_container_width=True,
                ):
                    st.session_state["mastishka_context_id"] = chat_context_id
                    st.session_state["mastishka_chat"] = []
                    st.rerun()

    if st.session_state.get("mastishka_context_id") != chat_context_id:
        st.session_state["mastishka_context_id"] = chat_context_id
        st.session_state.pop("mastishka_chat", None)
    st.session_state.setdefault(
        "mastishka_chat",
        [
            {
                "role": "assistant",
                "content": (
                    "Hi, I'm **Mastishka** 🧠 — your NeuroScan AI assistant. Ask me anything about "
                    "brain tumors, MRI scans, symptoms, treatments, or your latest scan result. "
                    "I combine knowledge from NCI & NINDS datasets with conversational AI."
                ),
                "retrieved_facts": [],
            }
        ],
    )

    allow_online = st.checkbox(
        "Use online AI and medical web search for answers",
        key=f"mastishka_online_consent_{chat_context_id}",
        help=(
            "When enabled, your text question and recent chat context are sent to the "
            "configured AI/search providers. Detected patient IDs, email addresses, and "
            "phone numbers are removed. Do not include identifying or sensitive details."
        ),
    )
    if allow_online:
        if not get_llm_client().is_available():
            st.warning("Online AI is not configured. Configure LLM_PROVIDER and LLM_API_KEY to enable it.")
        elif not is_web_search_available():
            st.info("Online AI is enabled, but live web citations require a TAVILY_API_KEY.")

    quick_pick = None
    user_message = None
    send_message = False
    if doctor_layout or patient_layout:
        with conversation_column:
            with st.container(border=True):
                title_column, clear_column = st.columns([4, 1])
                title_column.subheader("Mastishka AI")
                title_column.caption(
                    "Clinical explanations anchored to the selected scan."
                    if doctor_layout
                    else "Understand model output and prepare questions for your doctor."
                )
                if clear_column.button(
                    "Clear conversation",
                    key=(
                        "mastishka_clear_conversation"
                        if doctor_layout
                        else "mastishka_patient_clear_conversation"
                    ),
                    use_container_width=True,
                ):
                    st.session_state["mastishka_chat"] = []
                    st.rerun()
                safety_class = "doctor-callout" if doctor_layout else "patient-warning"
                st.markdown(
                    f'<div class="{safety_class}">Informational only, not a diagnosis. '
                    'Do not use this conversation to make treatment decisions. '
                    'Discuss results with a qualified clinician.</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Suggested questions")
                quick_cols = st.columns(max(1, min(3, len(SUGGESTED_QUESTIONS))))
                for index, question in enumerate(SUGGESTED_QUESTIONS):
                    column = quick_cols[index % len(quick_cols)]
                    if column.button(
                        question,
                        use_container_width=True,
                        key=f"quick_{question}",
                    ):
                        quick_pick = question
                for turn in st.session_state["mastishka_chat"]:
                    with st.chat_message(turn["role"]):
                        st.markdown(turn["content"])
                        facts = turn.get("retrieved_facts", [])
                        if facts:
                            with st.expander("Retrieved knowledge sources"):
                                for fact in facts:
                                    st.markdown(
                                        f"**Category: {fact['category']}** · "
                                        f"*Source: {fact['source']}* "
                                        f"(Score: {fact.get('score', 0):.2f})\n\n"
                                        f"**Q:** {fact['question']}\n\n"
                                        f"**A:** {fact['answer']}"
                                    )
                with st.form(
                    "mastishka_doctor_prompt_form"
                    if doctor_layout
                    else "mastishka_patient_prompt_form",
                    clear_on_submit=True,
                ):
                    prompt_column, send_column = st.columns([5, 1], gap="small")
                    user_message = prompt_column.text_input(
                        "Ask about this scan or AI result",
                        placeholder=(
                            "Ask about this scan or AI result..."
                            if scan_context
                            else "Ask a general brain MRI question..."
                        ),
                        label_visibility="collapsed",
                    )
                    send_message = send_column.form_submit_button(
                        "Send",
                        type="primary",
                        use_container_width=True,
                    )
                st.caption(
                    "AI explanations can be incomplete or incorrect. No diagnosis or emergency guidance is provided."
                )
    else:
        _card_start()
        st.caption("Suggested Questions by Category")
        quick_cols = st.columns(len(SUGGESTED_QUESTIONS))
        for col, question in zip(quick_cols, SUGGESTED_QUESTIONS):
            if col.button(question, use_container_width=True, key=f"quick_{question}"):
                quick_pick = question
        _card_end()
        for turn in st.session_state["mastishka_chat"]:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                facts = turn.get("retrieved_facts", [])
                if facts:
                    with st.expander("📚 Retrieved Knowledge Sources"):
                        for fact in facts:
                            st.markdown(
                                f"**Category: {fact['category']}** | "
                                f"*Source: {fact['source']}* "
                                f"(Score: {fact.get('score', 0):.2f})\n\n"
                                f"**Q:** {fact['question']}\n\n"
                                f"**A:** {fact['answer']}"
                            )

    if not doctor_layout and not patient_layout:
        user_message = st.chat_input("Ask Mastishka a question...")
    pending_question = st.session_state.pop("mastishka_pending_question", None)
    workspace_layout = doctor_layout or patient_layout
    incoming = quick_pick or (
        user_message if send_message or not workspace_layout else None
    ) or pending_question

    if incoming:
        # Append user message
        st.session_state["mastishka_chat"].append({"role": "user", "content": incoming, "retrieved_facts": []})

        # Process with RAG & conversation history
        res_dict = mastishka_response_dict(
            message=incoming,
            chat_history=st.session_state["mastishka_chat"],
            scan_result=(
                scan_context
                if workspace_layout
                else st.session_state.get("last_scan_result")
            ),
            allow_online=allow_online,
        )

        reply_text = res_dict["answer"]
        category = res_dict.get("category")
        source = res_dict.get("source")

        # Format disclaimers & sources
        if category in {"Treatment", "Prognosis", "Symptoms", "Medical Guidance", "Risk Factors"} and "Disclaimer:" not in reply_text:
            reply_text += "\n\n*Disclaimer: This information is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.*"

        if source in {"NCI", "NINDS", "NCI, NINDS"} and "Sources:" not in reply_text and "Source:" not in reply_text:
            reply_text += f"\n\n*Sources: {source}*"

        st.session_state["mastishka_chat"].append({
            "role": "assistant",
            "content": reply_text,
            "retrieved_facts": res_dict.get("retrieved_facts", []),
        })
        st.rerun()

    if not workspace_layout and st.button("Clear conversation"):
        st.session_state["mastishka_chat"] = []
        st.rerun()


def _pending_doctor_header(section: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="pending-workspace-marker"></div>
        <div class="pending-breadcrumb">Doctor Workspace&nbsp; / &nbsp;{html.escape(section)}</div>
        <h1 class="pending-page-title">{html.escape(title)}</h1>
        <div class="pending-page-subtitle">{html.escape(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


def _pending_doctor_user() -> dict:
    user_id = current_user_id()
    return get_user_by_id(user_id) if user_id else {}


def page_doctor_pending() -> None:
    user = _pending_doctor_user()
    name = _doctor_display_name(user)
    _pending_doctor_header(
        "Dashboard",
        f"Welcome, Dr. {name}",
        "Your account is set up. Your doctor verification is pending.",
    )
    st.markdown(
        '<div class="pending-status-badge">Status: Pending</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="pending-review-card">
          <span class="pending-pill">Pending</span>
          <h2>Your doctor profile is under review</h2>
          <p>Your credentials are awaiting administrator approval. You can access your profile
          and Mathiksha AI while you wait.</p>
          <div class="pending-divider"></div>
          <div class="pending-review-grid">
            <div><small>CURRENT STATUS</small><strong>Awaiting administrator approval</strong></div>
            <div><small>CLINICAL ACCESS</small><strong>Available after approval</strong></div>
          </div>
        </section>
        <h3 class="pending-section-title">Available while you wait</h3>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            '<div class="pending-feature-card pending-feature-highlight"><h3>Mathiksha AI</h3>'
            '<p>Explore your AI assistant while your doctor account is being reviewed.</p>',
            unsafe_allow_html=True,
        )
        if st.button("Open Mathiksha AI", key="pending_open_mastishka", type="primary"):
            st.session_state["page"] = "doctor_pending_mastishka"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div class="pending-feature-card"><h3>Your profile</h3>'
            '<p>Review your personal details and professional information.</p>',
            unsafe_allow_html=True,
        )
        if st.button("View profile", key="pending_open_profile"):
            st.session_state["page"] = "doctor_pending_profile"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        '<p class="pending-footnote">Patient management, scans, and clinical reports will be available once your account is approved.</p>',
        unsafe_allow_html=True,
    )


def page_doctor_pending_profile() -> None:
    user = _pending_doctor_user()
    _pending_doctor_header("Profile", "Your profile", "Manage your details and follow your verification status.")
    if st.button("Save changes", key="pending_profile_save", type="primary"):
        st.success("Your profile is already saved. Professional changes may require another review.")
    left, right = st.columns([1.7, 0.8], gap="medium")
    with left:
        st.markdown('<div class="pending-panel"><h3>Personal details</h3>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        p1.text_input("Full name", value=_doctor_display_name(user), disabled=True)
        p2.text_input("Email address", value=user.get("email") or "", disabled=True)
        p1.text_input("Phone number", value=user.get("phone") or "Not provided", disabled=True)
        p2.text_input("Account role", value="Doctor", disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="pending-panel"><h3>Professional information</h3>'
                    '<p class="pending-muted">These details are part of your verification review.</p>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        p1.text_input("Specialty", value=user.get("specialization") or "Not provided", disabled=True)
        p2.text_input("Doctor ID", value=f"NS-D-{int(user.get('id') or 0):03d}", disabled=True)
        p1.text_input("Medical registration number", value=user.get("license_no") or "Submitted for review", disabled=True)
        p2.text_input("Affiliated institution", value=user.get("hospital") or "Not provided", disabled=True)
        st.caption("Changes to professional details may require another review.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="pending-panel"><h3>Doctor verification</h3><span class="pending-pill">Pending</span>'
                    '<h4>Awaiting administrator approval</h4><p>Your account has limited access until your credentials are approved.</p>'
                    '<div class="pending-divider"></div><small>AVAILABLE NOW</small><strong>Dashboard · Mathiksha AI · Profile</strong></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="pending-panel"><h3>Submitted credentials</h3><p>Your professional credentials are awaiting administrator review.</p>'
                    '<strong class="pending-review-text">Review status: Pending</strong></div>', unsafe_allow_html=True)


def page_doctor_pending_mastishka() -> None:
    _pending_doctor_header("Mathiksha AI", "Mathiksha AI", "Your AI assistant for learning and getting started.")
    st.markdown(
        '<div class="pending-banner">Pending account · General guidance only. Patient data and clinical tools unlock after approval.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.7, 0.8], gap="medium")
    with left:
        st.markdown('<div class="pending-panel pending-chat-panel"><h3>Getting started</h3>'
                    '<p class="pending-muted">General information · No patient data</p><div class="pending-divider"></div>',
                    unsafe_allow_html=True)
        if "pending_chat" not in st.session_state:
            st.session_state["pending_chat"] = []
        for item in st.session_state["pending_chat"]:
            st.markdown(f'<div class="pending-chat-{item["role"]}">{html.escape(item["text"])}</div>', unsafe_allow_html=True)
        question = st.text_input("Ask a general question...", key="pending_chat_input", label_visibility="collapsed")
        if st.button("Send", key="pending_chat_send", type="primary") and question.strip():
            st.session_state["pending_chat"].append({"role": "user", "text": question.strip()})
            st.session_state["pending_chat"].append({"role": "assistant", "text": "While your account is under review, you can review your profile, ask general questions about the workspace, and learn how NeuroScan AI supports clinical workflows."})
            st.rerun()
        st.caption("Patient-specific advice, scans, and reports remain unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="pending-panel"><h3>Explore with Mathiksha</h3><p class="pending-muted">A few things you can ask</p>',
                    unsafe_allow_html=True)
        for question in ("How does verification work?", "What happens after approval?", "What is AI-assisted MRI review?"):
            if st.button(question, key=f"pending_question_{question}", use_container_width=True):
                st.session_state["pending_chat_input"] = question
                st.rerun()
        st.markdown("</div><div class=\"pending-panel pending-privacy\"><h3>Keep patient information private</h3><p>Do not enter names, medical records, or scan details in this general assistant.</p></div>", unsafe_allow_html=True)
def _render_doctor_scan_review(
    result: dict,
    prediction_id: int,
    target_patient: dict,
    patient_code: str,
    doctor_id: int,
    doctor_name: str,
) -> None:
    st.markdown(
        """
        <style>
        .clinical-review {color:#193d45;}
        .clinical-review h1,.clinical-review h2,.clinical-review h3 {color:#193d45;}
        .clinical-workflow {
          display:flex;align-items:center;justify-content:space-between;gap:8px;
          background:#fff;border:1px solid #e1eceb;border-radius:10px;
          padding:12px 16px;margin:12px 0 16px;
        }
        .clinical-workflow span {font-size:12px;color:#527078;}
        .clinical-workflow .active {font-weight:700;color:#138f86;}
        .clinical-panel {
          background:#fff;border:1px solid #e1eceb;border-radius:12px;
          padding:16px;height:100%;box-shadow:0 3px 14px rgba(25,61,69,.05);
        }
        .clinical-panel-title {font-size:14px;font-weight:700;margin-bottom:10px;color:#193d45;}
        .clinical-review-note {font-size:11px;color:#648087;margin-top:8px;}
        .clinical-disclaimer {
          background:#e5f5f3;border-radius:8px;padding:12px 14px;
          color:#315d62;font-size:11px;margin:14px 0;
        }
        </style>
        <div class="clinical-review">
          <div style="font-size:11px;color:#648087;">Doctor Workspace &nbsp; / &nbsp; Scan Patient</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title(f"Clinical review · {target_patient['username']}")
    prediction = get_prediction_by_id(prediction_id)
    created_at = prediction.get("created_at", "") if prediction else ""
    st.caption(
        f"{patient_code} · Scan #{prediction_id:06d}"
        + (f" · {created_at}" if created_at else "")
        + f" · {doctor_name}"
    )
    st.markdown(
        """
        <div class="clinical-workflow">
          <span>✓ &nbsp; Select patient</span>
          <span>✓ &nbsp; Upload &amp; qualify</span>
          <span>✓ &nbsp; AI analysis</span>
          <span class="active">4 &nbsp; Clinical review</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    confidence_pct = result["confidence"] * 100
    prediction_col, probabilities_col, context_col = st.columns([1, 1, 1.05], gap="medium")
    with prediction_col:
        with st.container(border=True):
            st.subheader("AI prediction")
            st.caption("AI-supported · clinician review required")
            st.markdown(f"### {result['class']} · {confidence_pct:.1f}%")
            st.caption(
                f"{result.get('model_name', 'AI model')} · {result.get('model_version', 'current version')}"
            )
            st.caption(f"Image quality: {result['quality']['label']}")
            if result.get("is_uncertain") or result.get("status") == "low_confidence":
                st.warning(result.get("low_confidence_warning", "Model output has limited confidence; review carefully."))

    with probabilities_col:
        with st.container(border=True):
            st.subheader("Class probabilities")
            for class_name, probability in result["probabilities"].items():
                st.caption(f"{class_name} · {probability * 100:.1f}%")
                st.progress(min(max(float(probability), 0.0), 1.0))

    with context_col:
        with st.container(border=True):
            st.subheader("Patient context")
            st.write(f"**{target_patient['username']}**")
            st.caption(f"Patient code: {patient_code}")
            st.caption(f"Email: {target_patient.get('email', 'Not provided')}")
            st.caption(f"Reviewing clinician: {doctor_name}")
            if st.button("Open full patient record", key=f"review_patient_{prediction_id}"):
                st.session_state.pop(f"last_doc_scan_res_{target_patient['id']}", None)
                st.rerun()

    steps = result.get("steps", {})
    segmentation = result.get("segmentation") or {}
    with st.container(border=True):
        st.subheader("MRI visualizations")
        image_panels = [
            ("Original", steps.get("original"), "Uploaded MRI image"),
            ("Preprocessed", steps.get("resized"), "Resized for model input"),
            ("Grad-CAM", result.get("gradcam"), "AI attention visualization"),
            ("Segmentation", segmentation.get("overlay"), "Illustrative region overlay"),
        ]
        image_columns = st.columns(4, gap="small")
        for column, (label, image, caption) in zip(image_columns, image_panels):
            with column:
                st.caption(label)
                if image is not None:
                    st.image(image, use_column_width=True)
                    st.caption(caption)
                else:
                    st.info(f"{label} is unavailable for this scan.")

    with st.container(border=True):
        st.subheader("AI summary")
        st.write(result.get("ai_summary", "No AI summary is available for this scan."))
        st.caption(
            "AI-generated findings support discussion only. A qualified clinician must review the scan and complete MRI study."
        )
        if segmentation.get("disclaimer"):
            st.caption(segmentation["disclaimer"])

    with st.container(border=True):
        st.subheader("Independent clinical review")
        st.caption("Clinical notes are stored with this scan and are visible in the patient record.")
    existing_notes = get_clinical_notes_for_prediction(prediction_id)
    for note in existing_notes:
        with st.container(border=True):
            st.caption(f"Reviewed by {note['doctor_name']} · {note['created_at']}")
            st.write(note["note_content"])
            if note.get("recommendation"):
                st.write(f"**Recommendation:** {note['recommendation']}")

    with st.form(f"doc_scan_note_form_{prediction_id}"):
        note_text = st.text_area(
            "Clinical observation",
            placeholder="Document your independent review and clinical context.",
        )
        recommendation = st.text_area(
            "Clinical recommendation",
            placeholder="Record follow-up, additional imaging or referral recommendations.",
        )
        acknowledged = st.checkbox(
            "I understand the AI output is illustrative and does not replace clinical judgment."
        )
        save_note = st.form_submit_button("Save clinical note")

    if save_note:
        if not note_text.strip():
            st.error("Enter a clinical observation before saving.")
        elif not acknowledged:
            st.error("Please acknowledge the clinical review disclaimer before saving.")
        else:
            add_clinical_note(
                prediction_id,
                doctor_id,
                note_text.strip(),
                recommendation.strip() or None,
            )
            st.success("Clinical note saved to the patient record.")
            st.rerun()

    clinician_recommendations = [
        note["recommendation"]
        for note in existing_notes
        if note.get("recommendation")
    ] or ["No clinician recommendation has been recorded for this scan."]
    report_ai = _report_ai_fields(
        result["class"],
        float(result["confidence"]),
        result["probabilities"],
        result.get("ai_summary"),
    )
    pdf_bytes = generate_pdf_report(
        username=target_patient["username"],
        scan_date=str(created_at or datetime.now().strftime("%d %B %Y %H:%M:%S")),
        original_img=steps["original"],
        preprocessed_img=steps["resized"],
        gradcam_img=result.get("gradcam"),
        segmentation_img=segmentation.get("overlay"),
        prediction=result["class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        class_info={**CLASS_INFO[result["class"]], "severity": result["severity"]},
        recommendations=clinician_recommendations,
        clinical_notes=existing_notes,
        scan_id=prediction_id,
        ai_summary=report_ai["summary"],
        ai_sources=report_ai["sources"],
        research_status=report_ai["status"],
        model_name=result.get("model_name"),
        quality_label=result["quality"]["label"],
    )
    st.download_button(
        label="Download PDF report",
        data=pdf_bytes,
        file_name=f"NeuroScan_Report_{patient_code}_{prediction_id}.pdf",
        mime="application/pdf",
        key=f"pdf_dl_doc_{prediction_id}",
    )


def page_doctor_dashboard() -> None:
    doctor_id = _doctor_page_access()
    if not doctor_id:
        return

    doc_user = get_user_by_id(doctor_id) or {}
    doctor_name = _doctor_display_name(doc_user)
    _doctor_workspace_header(
        "Clinical Dashboard",
        f"Good morning, Dr. {doctor_name}",
        "Review imaging, add clinical context and guide the next step.",
    )
    assigned_patients = get_doctor_assigned_patients_with_details(doctor_id)
    assigned_predictions = get_doctor_assigned_predictions(doctor_id)
    invitations = get_doctor_pending_invitations(doctor_id)
    doctor_appointments = get_doctor_appointments(doctor_id)
    scheduled_appointments = [
        appointment for appointment in doctor_appointments
        if appointment["status"] == "scheduled"
        and datetime.fromisoformat(appointment["scheduled_at"]) >= datetime.now()
    ]
    notes_by_prediction = {
        prediction["id"]: get_clinical_notes_for_prediction(prediction["id"])
        for prediction in assigned_predictions
    }
    signed_notes = sum(len(notes) for notes in notes_by_prediction.values())
    awaiting_review = sum(not notes for notes in notes_by_prediction.values())

    metrics = st.columns(4, gap="small")
    metrics[0].metric("My patients", len(assigned_patients))
    metrics[1].metric("Awaiting review", f"{awaiting_review:02d}")
    metrics[2].metric("Notes saved", signed_notes)
    metrics[3].metric(
        "Consultations today",
        sum(
            datetime.fromisoformat(appointment["scheduled_at"]).date() == date.today()
            for appointment in scheduled_appointments
        ),
    )
    with st.container(border=True):
        st.subheader("Upcoming consultations")
        if scheduled_appointments:
            for appointment in scheduled_appointments[:5]:
                scheduled = datetime.fromisoformat(appointment["scheduled_at"])
                st.markdown(
                    f"**{scheduled.strftime('%d %b · %I:%M %p')}** · "
                    f"{html.escape(appointment['patient_name'])} · "
                    f"{html.escape(appointment['appointment_type'])}"
                )
                st.caption(
                    f"{html.escape(appointment['reason'])} · "
                    f"{html.escape(appointment.get('patient_email') or '')}"
                )
        else:
            st.info("No upcoming consultations are scheduled.")

    patient_search, patient_action = st.columns([4, 1], gap="small")
    search_text = patient_search.text_input(
        "Search patients by name or patient ID",
        placeholder="Search patients by name or patient ID...",
        key="dashboard_patient_search",
        label_visibility="collapsed",
    ).strip().lower()
    visible_patients = [
        item
        for item in assigned_patients
        if not search_text
        or search_text
        in f"{item['full_name']} {item['patient_code']} {item.get('email', '')}".lower()
    ]
    if patient_action.button(
        "Invite patient",
        use_container_width=True,
        key="dashboard_establish_relationship",
    ):
        st.session_state["page"] = "doctor_add_patient"
        st.rerun()

    selected = None
    with st.container():
        if visible_patients:
            labels = {
                f"{item['full_name']} · {item['patient_code']}": item
                for item in visible_patients
            }
            labels_list = list(labels)
            saved_patient_id = st.session_state.get("dashboard_selected_patient_id")
            selected_index = next(
                (
                    index for index, label in enumerate(labels_list)
                    if labels[label]["id"] == saved_patient_id
                ),
                0,
            )
            patient_label = st.selectbox(
                "Select a patient before uploading an MRI",
                labels_list,
                index=selected_index,
                key="dashboard_patient_select",
            )
            selected = labels.get(patient_label) if patient_label else None
            if selected:
                st.session_state["dashboard_selected_patient_id"] = selected["id"]
                profile_col, details_col, action_col = st.columns([0.6, 4, 1.2])
                profile_col.markdown(
                    f'<div class="doctor-patient-initial">{html.escape(selected["full_name"][:1].upper())}</div>',
                    unsafe_allow_html=True,
                )
                patient_scans = get_user_predictions(selected["id"], limit=100)
                details_col.markdown(f"**{html.escape(selected['full_name'])}**")
                details_col.caption(
                    f"{selected['patient_code']} · {selected.get('dob') or 'DOB not recorded'} · "
                    f"{selected.get('gender') or 'Sex not recorded'} · "
                    f"{len(patient_scans)} scans · Assigned {selected.get('assigned_at') or 'date not recorded'}"
                )
                if action_col.button(
                    "Patient profile",
                    key="dashboard_patient_profile",
                    use_container_width=True,
                ):
                    st.session_state["selected_doctor_patient_id"] = selected["id"]
                    st.session_state["page"] = "doctor_patients"
                    st.rerun()
        else:
            st.info(
                "No patients match the current search."
                if assigned_patients
                else "No accepted patient relationships yet. Invite a patient and wait for acceptance before clinical access."
            )

    if selected:
        patient_scans = get_user_predictions(selected["id"], limit=100)
        with st.container(border=True):
            model_available = is_model_available()
            if not model_available:
                st.warning(
                    "MRI predictions are disabled because the trained model file is not available."
                )
            upload_info, uploader, upload_action = st.columns([2.4, 3.2, 1.2])
            upload_info.markdown(f"**Upload MRI for {html.escape(selected['full_name'])}**")
            upload_info.caption("JPG or PNG · up to 10 MB")
            uploaded_file = uploader.file_uploader(
                "Browse files",
                type=["jpg", "jpeg", "png"],
                key=f"dashboard_upload_{selected['id']}",
                label_visibility="collapsed",
            )
            if upload_action.button(
                "Analyze MRI",
                type="primary",
                key=f"dashboard_analyze_{selected['id']}",
                use_container_width=True,
                disabled=uploaded_file is None or not model_available,
            ):
                allowed, message = can_doctor_scan_patient(doctor_id, selected["id"])
                if not allowed:
                    st.error(message)
                else:
                    valid_file, validation_message = validate_image(uploaded_file)
                    if not valid_file:
                        st.error(validation_message)
                    else:
                        with st.spinner("Analyzing the uploaded MRI..."):
                            result = analyze_uploaded_scan(uploaded_file)
                        prediction_id = _save_scan_result(
                            result,
                            uploaded_file,
                            target_user_id=selected["id"],
                            doctor_id=doctor_id,
                        )
                        log_audit_event(
                            user_id=doctor_id,
                            action="DOCTOR_PERFORMED_PATIENT_SCAN",
                            resource=f"patient:{selected['id']}",
                            details={
                                "prediction_id": prediction_id,
                                "class": result["class"],
                            },
                        )
                        st.session_state[f"last_doc_scan_res_{selected['id']}"] = (
                            result,
                            prediction_id,
                        )
                        st.rerun()

        if patient_scans:
            saved_scan_id = st.session_state.get("dashboard_selected_scan_id")
            selected_scan_index = next(
                (
                    index for index, scan in enumerate(patient_scans)
                    if scan["id"] == saved_scan_id
                ),
                0,
            )
            scan_labels = [
                f"NS-{scan['id']:06d} · {scan['created_at']} · {scan['prediction']}"
                for scan in patient_scans
            ]
            selected_scan_label = st.selectbox(
                "Selected MRI scan",
                scan_labels,
                index=selected_scan_index,
                key="dashboard_selected_scan",
            )
            selected_scan_index = (
                scan_labels.index(selected_scan_label) if selected_scan_label in scan_labels else 0
            )
            selected_scan = patient_scans[selected_scan_index]
            st.session_state["dashboard_selected_scan_id"] = selected_scan["id"]
            record = get_prediction_by_id(selected_scan["id"])
            result_key = f"last_doc_scan_res_{selected['id']}"
            cached_scan = st.session_state.get(result_key)
            analysis_result = (
                cached_scan[0]
                if cached_scan and cached_scan[1] == selected_scan["id"]
                else None
            )
            notes = get_clinical_notes_for_prediction(selected_scan["id"])
            scan_steps = analysis_result.get("steps") if analysis_result else None
            gradcam_image = analysis_result.get("gradcam") if analysis_result else None
            segmentation = analysis_result.get("segmentation") if analysis_result else None
            if not scan_steps and record and record.get("image_data"):
                try:
                    original_image = np.asarray(
                        Image.open(io.BytesIO(record["image_data"])).convert("RGB")
                    )
                except (UnidentifiedImageError, OSError):
                    st.warning("The stored scan image could not be opened.")
                else:
                    scan_steps = preprocess_array(original_image)

            if analysis_result:
                probabilities = analysis_result["probabilities"]
                summary_text = analysis_result.get("ai_summary")
                quality_label = analysis_result.get("quality", {}).get("label")
                prediction = analysis_result["class"]
                confidence = float(analysis_result["confidence"])
                severity = analysis_result.get("severity")
                model_name = analysis_result.get("model_name")
            elif record:
                probabilities = {
                    "Glioma": float(record.get("glioma_prob") or 0),
                    "Meningioma": float(record.get("meningioma_prob") or 0),
                    "Pituitary": float(record.get("pituitary_prob") or 0),
                    "Normal": float(record.get("normal_prob") or 0),
                }
                summary_text = record.get("ai_summary")
                quality_label = record.get("quality_label")
                prediction = record["prediction"]
                confidence = float(record["confidence"])
                severity = record.get("severity")
                model_name = record.get("model_name")
            else:
                probabilities = {}
                summary_text = None
                quality_label = None
                prediction = selected_scan["prediction"]
                confidence = float(selected_scan["confidence"])
                severity = selected_scan.get("severity")
                model_name = selected_scan.get("model_name")

            with st.container(border=True):
                title_col, badge_col = st.columns([4, 1])
                title_col.subheader("Review the scan. Add the clinical context.")
                title_col.caption(
                    f"{selected['full_name']} · NS-{selected_scan['id']:06d} · {selected_scan['created_at']}"
                )
                badge_col.caption("CLINICAL REVIEW")
                workflow = st.columns(5)
                for column, step in zip(
                    workflow,
                    ["Patient selected", "MRI uploaded", "AI analysis", "Clinical review", "Note & PDF"],
                ):
                    column.caption(f"✓ {step}" if step != "Clinical review" else f"● {step}")

                image_column, analysis_column = st.columns([2.25, 1], gap="medium")
                image_panels = [
                    ("Original", scan_steps.get("original") if scan_steps else None),
                    ("Preprocessed", scan_steps.get("resized") if scan_steps else None),
                    ("Grad-CAM", gradcam_image),
                    (
                        "Segmentation",
                        (segmentation or {}).get("overlay") if isinstance(segmentation, dict) else None,
                    ),
                ]
                with image_column:
                    image_rows = [st.columns(2, gap="small"), st.columns(2, gap="small")]
                    for index, (label, image) in enumerate(image_panels):
                        tile = image_rows[index // 2][index % 2]
                        with tile:
                            st.caption(label)
                            if image is not None:
                                st.image(image, use_column_width=True)
                            elif label in {"Grad-CAM", "Segmentation"}:
                                st.caption("Not stored for this earlier scan.")
                            else:
                                st.info("MRI image unavailable.")
                with analysis_column:
                    st.caption("AI prediction · decision support")
                    st.subheader(prediction)
                    st.metric("Model confidence", f"{confidence * 100:.1f}%")
                    if model_name:
                        st.caption(model_name)
                    if quality_label:
                        st.caption(f"MRI quality · {quality_label}")
                    if probabilities:
                        for class_name, value in probabilities.items():
                            st.caption(f"{class_name} · {float(value) * 100:.1f}%")
                            st.progress(min(max(float(value), 0.0), 1.0))
                    if summary_text:
                        st.info(summary_text)
                    st.caption(
                        "Confidence describes model output; it is not a diagnosis or a measure of tumour growth."
                    )
                st.warning(
                    "AI predictions require independent clinical review. Model confidence is not a diagnosis."
                )

            lower_left, lower_right = st.columns([2.1, 1], gap="small")
            with lower_left:
                with st.container(border=True):
                    st.subheader("Your clinical note")
                    st.caption(
                        f"Attached to {selected['full_name']} · NS-{selected_scan['id']:06d}"
                    )
                    for note in notes:
                        with st.container(border=True):
                            st.caption(f"Saved by Dr. {note['doctor_name']} · {note['created_at']}")
                            st.write(note["note_content"])
                            if note.get("recommendation"):
                                st.write(f"**Recommendation:** {note['recommendation']}")
                    with st.form(f"dashboard_note_form_{selected_scan['id']}"):
                        note_text = st.text_area(
                            "Clinical observation",
                            placeholder="Document your independent review.",
                        )
                        recommendation = st.text_area(
                            "Recommendation",
                            placeholder="Record follow-up or care-plan recommendations.",
                        )
                        acknowledged = st.checkbox(
                            "I reviewed the source scan; AI output is not a diagnosis."
                        )
                        save_note = st.form_submit_button("Save clinical note", type="primary")
                    if save_note:
                        if not note_text.strip():
                            st.error("Enter a clinical observation before saving.")
                        elif not acknowledged:
                            st.error("Acknowledge the clinical review before saving.")
                        else:
                            add_clinical_note(
                                selected_scan["id"],
                                doctor_id,
                                note_text.strip(),
                                recommendation.strip() or None,
                            )
                            st.success("Clinical note saved to the patient record.")
                            st.rerun()

                    if record and scan_steps:
                        pdf_recommendations = [
                            note["recommendation"]
                            for note in notes
                            if note.get("recommendation")
                        ] or ["No clinician recommendation has been recorded for this scan."]
                        report_ai = _report_ai_fields(
                            prediction,
                            confidence,
                            probabilities,
                            summary_text,
                        )
                        pdf = generate_pdf_report(
                            username=selected["full_name"],
                            scan_date=str(record.get("created_at") or ""),
                            original_img=scan_steps["original"],
                            preprocessed_img=scan_steps["resized"],
                            gradcam_img=gradcam_image,
                            segmentation_img=(segmentation or {}).get("overlay")
                            if isinstance(segmentation, dict)
                            else None,
                            prediction=prediction,
                            confidence=confidence,
                            probabilities=probabilities,
                            class_info={
                                **CLASS_INFO.get(prediction, {}),
                                "severity": severity,
                            },
                            recommendations=pdf_recommendations,
                            clinical_notes=notes,
                            scan_id=selected_scan["id"],
                            ai_summary=report_ai["summary"],
                            ai_sources=report_ai["sources"],
                            research_status=report_ai["status"],
                            model_name=model_name,
                            quality_label=quality_label,
                        )
                        st.download_button(
                            "Preview / download PDF",
                            data=pdf,
                            file_name=f"NeuroScan_Report_{selected_scan['id']:06d}.pdf",
                            mime="application/pdf",
                            key=f"dashboard_pdf_{selected_scan['id']}",
                        )
                    else:
                        st.caption("A PDF preview is unavailable because the original scan image is missing.")
                    st.caption("Saving a note does not confirm the AI prediction or constitute a digital signature.")

            with lower_right:
                with st.container(border=True):
                    st.subheader("Training Hub")
                    st.caption("Saved model evaluation · evaluation data only")
                    metrics_path = Path("logs/evaluation_metrics.json")
                    if not metrics_path.exists():
                        metrics_path = Path("data/artifacts/plots/evaluation_metrics.json")
                    if metrics_path.exists():
                        evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
                        evaluation_rows = [
                            {"Metric": "Accuracy", "Value": f"{float(evaluation['accuracy']) * 100:.1f}%"},
                            {
                                "Metric": "Macro F1",
                                "Value": f"{float(evaluation['macro_avg']['f1_score']):.3f}",
                            },
                            {
                                "Metric": "Evaluation samples",
                                "Value": str(evaluation.get("total_samples", "Not recorded")),
                            },
                        ]
                        st.dataframe(
                            pd.DataFrame(evaluation_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No model evaluation snapshot is available.")
                    if st.button("Open Training Hub ↗", key="dashboard_training_hub"):
                        st.session_state["page"] = "training"
                        st.rerun()

            history_col, ai_col = st.columns([2.1, 1], gap="small")
            with history_col:
                with st.container(border=True):
                    st.subheader("Patient history & analytics")
                    st.caption(f"{selected['full_name']} · Longitudinal imaging context")
                    all_notes = sum(
                        len(get_clinical_notes_for_prediction(scan["id"]))
                        for scan in patient_scans
                    )
                    latest_quality = patient_scans[0].get("quality_label") if patient_scans else None
                    history_metrics = st.columns(3)
                    history_metrics[0].metric("MRI scans", len(patient_scans))
                    history_metrics[1].metric("Clinical notes", all_notes)
                    history_metrics[2].metric("Latest MRI quality", latest_quality or "Not assessed")
                    history_rows = []
                    for scan in patient_scans:
                        scan_notes = get_clinical_notes_for_prediction(scan["id"])
                        history_rows.append(
                            {
                                "Scan date": scan["created_at"],
                                "Scan ID": f"NS-{scan['id']:06d}",
                                "AI output / score": (
                                    f"{scan['prediction']} · {float(scan['confidence']) * 100:.1f}%"
                                ),
                                "Review status": "Clinical notes saved" if scan_notes else "Awaiting review",
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(history_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "Changes in model confidence do not independently establish tumour growth or clinical progression."
                    )

            with ai_col:
                with st.container(border=True):
                    st.subheader("Mastishka AI")
                    st.caption("Understand the selected scan")
                    st.info(
                        "Ask for an explanation of the model output or its limitations. Responses are informational, not diagnostic."
                    )
                    if st.button(
                        "Explain this scan ↗",
                        key="dashboard_mastishka_explain",
                        use_container_width=True,
                    ):
                        st.session_state["mastishka_patient_context"] = selected
                        st.session_state["mastishka_scan_context"] = patient_scans[
                            selected_scan_index
                        ]
                        st.session_state["mastishka_pending_question"] = (
                            "Explain the limitations of this MRI model output."
                        )
                        st.session_state["page"] = "mastishka"
                        st.rerun()
                    if st.button(
                        "Review model limitations",
                        key="dashboard_mastishka_limits",
                        use_container_width=True,
                    ):
                        st.session_state["mastishka_patient_context"] = selected
                        st.session_state["mastishka_scan_context"] = patient_scans[
                            selected_scan_index
                        ]
                        st.session_state["mastishka_pending_question"] = (
                            "What are the limitations of AI-assisted MRI analysis?"
                        )
                        st.session_state["page"] = "mastishka"
                        st.rerun()
                    st.caption("Always discuss results with a qualified clinician.")
        else:
            with st.container(border=True):
                st.info("No MRI scans are recorded for this patient yet. Upload a scan above to begin a review.")
    else:
        empty_message = (
            "No patients match this search."
            if assigned_patients
            else "No accepted patient relationship is available yet."
        )
        with st.container(border=True):
            st.subheader("MRI upload")
            st.caption(empty_message + " Uploads are available after a patient relationship is accepted.")
            upload_col, review_col = st.columns([1, 2], gap="medium")
            with upload_col:
                st.file_uploader(
                    "Upload patient MRI",
                    type=["jpg", "jpeg", "png"],
                    disabled=True,
                    key="dashboard_empty_upload",
                )
                st.caption("JPG or PNG · up to 10 MB")
                st.button(
                    "Analyze MRI",
                    type="primary",
                    disabled=True,
                    key="dashboard_empty_analyze",
                    use_container_width=True,
                )
            with review_col:
                st.subheader("Scan review")
                image_rows = [st.columns(2, gap="small"), st.columns(2, gap="small")]
                for index, label in enumerate(
                    ["Original", "Preprocessed", "Grad-CAM", "Segmentation"]
                ):
                    column = image_rows[index // 2][index % 2]
                    with column:
                        st.caption(label)
                        st.caption("Available after an authorized scan is selected.")
                st.caption("AI prediction and confidence will appear after analysis.")

        lower_left, lower_right = st.columns([2.1, 1], gap="small")
        with lower_left:
            with st.container(border=True):
                st.subheader("Your clinical note")
                st.caption("Select an accepted patient and scan before adding clinical notes.")
                st.text_area(
                    "Clinical observation",
                    placeholder="Document your independent review.",
                    disabled=True,
                    key="dashboard_empty_note",
                )
                st.text_area(
                    "Recommendation",
                    placeholder="Record follow-up or care-plan recommendations.",
                    disabled=True,
                    key="dashboard_empty_recommendation",
                )
                st.button(
                    "Save clinical note",
                    type="primary",
                    disabled=True,
                    key="dashboard_empty_save_note",
                )
        with lower_right:
            with st.container(border=True):
                st.subheader("Training Hub")
                st.caption("Explore saved model evaluation results.")
                if st.button("Open Training Hub ↗", key="dashboard_empty_training"):
                    st.session_state["page"] = "training"
                    st.rerun()

        history_col, ai_col = st.columns([2.1, 1], gap="small")
        with history_col:
            with st.container(border=True):
                st.subheader("Patient history & analytics")
                st.caption("Patient history becomes available after consent.")
        with ai_col:
            with st.container(border=True):
                st.subheader("Mastishka AI")
                st.caption("Ask general questions about MRI analysis and its limitations.")
                if st.button(
                    "Open Mastishka AI ↗",
                    key="dashboard_empty_mastishka",
                    use_container_width=True,
                ):
                    st.session_state["mastishka_pending_question"] = (
                        "What are the limitations of AI-assisted MRI analysis?"
                    )
                    st.session_state["page"] = "mastishka"
                    st.rerun()

    if invitations:
        with st.container(border=True):
            st.subheader("Relationships awaiting patient acceptance")
            for invitation in invitations:
                st.write(
                    f"**{invitation['patient_name']}** · {invitation['relationship_role']} · "
                    f"Invitation expires {invitation['expires_at']}"
                )
                st.caption(
                    "Patient scans and reports stay private until the patient accepts this invitation."
                )


def page_doctor_patients() -> None:
    doctor_id = _doctor_page_access()
    if not doctor_id:
        return
    _doctor_workspace_header(
        "My Patients",
        "Every patient, in context",
        "Manage care relationships, review imaging history and focus clinical work.",
    )
    patients = get_doctor_assigned_patients_with_details(doctor_id)
    predictions = get_doctor_assigned_predictions(doctor_id)
    notes_count = sum(
        len(get_clinical_notes_for_prediction(prediction["id"]))
        for prediction in predictions
    )
    pending = get_doctor_pending_invitations(doctor_id)
    metric_cols = st.columns(4)
    metric_cols[0].metric("My patients", len(patients))
    metric_cols[1].metric("Awaiting review", sum(
        not get_clinical_notes_for_prediction(prediction["id"])
        for prediction in predictions
    ))
    metric_cols[2].metric("Notes signed", notes_count)
    metric_cols[3].metric("Pending consent", len(pending))

    filter_bar = st.columns([2.4, 1, 1])
    query = filter_bar[0].text_input(
        "Search patients by name, patient ID or email",
        placeholder="Search patient directory",
        key="doctor_patient_search",
        label_visibility="collapsed",
    )
    query_lower = query.strip().lower()
    visible = [
        patient for patient in patients
        if not query_lower or query_lower in " ".join(
            str(patient.get(key, "")) for key in ("full_name", "email", "phone", "patient_code")
        ).lower()
    ]
    review_filter = filter_bar[1].selectbox(
        "Review status",
        ["All reviews", "Awaiting review", "Clinical notes saved"],
        label_visibility="collapsed",
        key="patient_review_filter",
    )
    if review_filter != "All reviews":
        filtered_patients = []
        for item in visible:
            scans_for_filter = get_user_predictions(item["id"], limit=100)
            note_states = [
                bool(get_clinical_notes_for_prediction(scan["id"]))
                for scan in scans_for_filter
            ]
            has_pending = any(not has_notes for has_notes in note_states)
            has_signed = any(note_states)
            if (review_filter == "Awaiting review" and has_pending) or (
                review_filter == "Clinical notes saved" and has_signed
            ):
                filtered_patients.append(item)
        visible = filtered_patients
    if filter_bar[2].button(
        "＋ Add Patient",
        key="patient_directory_add",
        use_container_width=True,
        type="primary",
    ):
        st.session_state["page"] = "doctor_add_patient"
        st.rerun()

    if not visible:
        with st.container(border=True):
            st.subheader("Patient directory")
            st.caption(
                f"{len(patients)} accepted relationships · {len(pending)} invitations awaiting consent"
            )
            st.info(
                "No patients match this view. Invite a patient and wait for them to accept before opening their records."
            )
        return

    rows = []
    for patient in visible:
        scans_for_patient = get_user_predictions(patient["id"], limit=100)
        note_count = sum(
            len(get_clinical_notes_for_prediction(scan["id"]))
            for scan in scans_for_patient
        )
        latest = scans_for_patient[0] if scans_for_patient else None
        awaiting = bool(latest and not get_clinical_notes_for_prediction(latest["id"]))
        rows.append({
            "Patient": patient["full_name"],
            "Patient ID": patient["patient_code"],
            "DOB / sex": f"{patient.get('dob') or '—'} · {patient.get('gender') or '—'}",
            "Latest MRI": f"{latest['created_at']} · {len(scans_for_patient)} scans" if latest else "No scans",
            "Review status": "Awaiting review" if awaiting else ("Clinical notes saved" if note_count else "No review yet"),
        })
    with st.container(border=True):
        st.subheader("Patient directory")
        st.caption(
            f"{len(visible)} of {len(patients)} accepted relationships · "
            "Only accepted relationships allow clinical record access."
        )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        labels = {
            f"{patient['full_name']} · {patient['patient_code']}": patient
            for patient in visible
        }
        selected_label = st.selectbox(
            "Selected patient",
            list(labels),
            key="patients_selected_patient",
        )
        selected = labels.get(selected_label) if selected_label else None
    if not selected:
        with st.container(border=True):
            st.info("Select a patient to view their record.")
        return

    left, right = st.columns([2, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Selected patient")
            st.caption(f"{selected['full_name']} · {selected['patient_code']}")
            details = st.columns(3)
            details[0].write(f"**Email:** {selected['email']}")
            details[1].write(f"**Phone:** {selected.get('phone') or 'Not recorded'}")
            details[2].write(f"**Assigned:** {selected.get('assigned_at') or 'Not recorded'}")
            st.write(
                f"**Medical history:** {selected.get('medical_history') or 'No history recorded.'}"
            )
            actions = st.columns(3)
            if actions[0].button(
                "Scan patient",
                type="primary",
                key="selected_patient_scan",
                use_container_width=True,
            ):
                st.session_state["scanning_patient_id"] = selected["id"]
                st.session_state["page"] = "doctor_scan_patient"
                st.rerun()
            if actions[1].button(
                "View patient reports",
                key="selected_patient_reports",
                use_container_width=True,
            ):
                st.session_state["report_patient_filter"] = selected["full_name"]
                st.session_state["page"] = "doctor_reports"
                st.rerun()
            if actions[2].button(
                "Manage relationship",
                key="selected_patient_relationship",
                use_container_width=True,
            ):
                st.session_state["doctor_invite_patient_id"] = selected["id"]
                st.session_state["page"] = "doctor_add_patient"
                st.rerun()
            scans = get_user_predictions(selected["id"], limit=100)
            if scans:
                st.dataframe(pd.DataFrame(scans), use_container_width=True, hide_index=True)
            else:
                st.info("No scans on record.")
    with right:
        with st.container(border=True):
            st.subheader("Patient analytics")
            st.write(f"**Total MRI scans:** {len(scans)}")
            patient_note_count = sum(
                len(get_clinical_notes_for_prediction(scan["id"])) for scan in scans
            )
            st.write(f"**Signed clinical notes:** {patient_note_count}")
            quality_values = [
                scan.get("quality_label") for scan in scans if scan.get("quality_label")
            ]
            st.write(
                f"**Latest MRI quality:** {quality_values[0] if quality_values else 'Not assessed'}"
            )
            if scans:
                confidence = pd.DataFrame(scans)
                confidence["confidence"] = confidence["confidence"].astype(float) * 100
                st.line_chart(
                    confidence.sort_values("created_at").set_index("created_at")["confidence"],
                )
                st.caption("Model confidence (%)")
            st.caption(
                "Changes in model confidence do not independently establish tumour growth or clinical progression."
            )


def _deliver_patient_invitation(
    token: str, patient_email: str, doctor_id: int
) -> tuple[bool, str]:
    base_url = os.getenv("APP_BASE_URL", "http://localhost:8501").rstrip("/")
    invitation_url = f"{base_url}/?page=accept_invitation&token={token}"
    sent, message = send_patient_invitation(patient_email, invitation_url)
    if not sent:
        revoke_patient_invitation(token, doctor_id)
        return False, message
    log_audit_event(
        user_id=doctor_id,
        action="patient_relationship_invitation_sent",
        details={"recipient": patient_email},
    )
    return True, message


def page_doctor_add_patient() -> None:
    doctor_id = _doctor_page_access()
    if not doctor_id:
        return
    doctor = get_user_by_id(doctor_id) or {}
    _doctor_workspace_header(
        "Add Patient",
        "Create a patient account",
        "Register a patient directly, send their temporary credentials, and begin scanning immediately.",
    )
    with st.container(border=True):
        st.markdown("**Direct patient onboarding**")
        st.caption("The patient receives a structured email with a temporary reference password and can change it from Profile.")
    existing_tab, new_tab = st.tabs(["Find existing patient", "Register new patient"])
    with existing_tab:
        found = None
        with st.form("doctor_find_patient_form"):
            identifier = st.text_input(
                "Name, email, phone or patient ID",
                placeholder="Search patient account",
            )
            search = st.form_submit_button("Find existing patient")
        if search:
            if not identifier.strip():
                st.error("Enter a name, email, phone or patient ID.")
            else:
                found = find_patient_by_identifier(identifier)
                if found and found["id"] == doctor_id:
                    found = None
                    st.error("You cannot invite your own account.")
                elif found:
                    st.session_state["doctor_invite_patient_id"] = found["id"]
                else:
                    st.warning("No patient account matched that information.")

        patient_id = st.session_state.get("doctor_invite_patient_id")
        if patient_id:
            found = get_user_by_id(patient_id)
            if not found or found.get("role") != "patient":
                st.session_state.pop("doctor_invite_patient_id", None)
                st.error("The selected patient account is unavailable.")
            else:
                patient_code = f"PAT-{patient_id:06d}"
                with st.container(border=True):
                    st.markdown(
                        f"**{html.escape(found['username'])}** · {patient_code}  \n"
                        f"{html.escape(found['email'])}"
                    )
                if is_patient_assigned_to_doctor(patient_id, doctor_id):
                    st.success("This patient relationship is already accepted and active.")
                else:
                    with st.form("existing_patient_invitation_form"):
                        relationship_role = st.selectbox(
                            "Relationship role",
                            ["Primary doctor", "Consulting specialist", "Radiologist", "Other"],
                        )
                        institution = st.text_input(
                            "Care institution",
                            value=doctor.get("hospital") or "",
                        )
                        consent_note = st.text_area(
                            "Invitation note",
                            placeholder="Explain why you are requesting access to scans and reports.",
                        )
                        authorization_confirmed = st.checkbox(
                            "I am authorized to invite this patient. Their consent has not yet been granted."
                        )
                        send_existing = st.form_submit_button(
                            "Review and send invitation", type="primary"
                        )
                    if send_existing:
                        if not authorization_confirmed:
                            st.error("Confirm your authority to invite this patient.")
                        elif relationship_role is None:
                            st.error("Select a relationship role before sending the invitation.")
                        else:
                            ok, message, token = create_patient_relationship_invitation(
                                doctor_id,
                                patient_id,
                                relationship_role,
                                institution,
                                consent_note,
                            )
                            if ok and token:
                                sent, delivery_message = _deliver_patient_invitation(
                                    token, found["email"], doctor_id
                                )
                                if sent:
                                    st.success(message)
                                    if os.getenv("APP_ENV", "development").lower() != "production" and not (
                                        os.getenv("SMTP_USERNAME") and os.getenv("SMTP_PASSWORD")
                                    ):
                                        st.caption("Development mode: invitation link is available in the app server log.")
                                    st.session_state.pop("doctor_invite_patient_id", None)
                                else:
                                    st.error(f"Invitation was not sent: {delivery_message}")
                            else:
                                st.error(message)
    with new_tab:
        with st.form("doctor_create_patient_form"):
            first_name, last_name = st.columns(2)
            first = first_name.text_input("First name *")
            last = last_name.text_input("Last name *")
            dob = st.date_input("Date of birth *")
            gender = st.selectbox("Sex recorded for care", ["Prefer not to say", "Female", "Male", "Other"])
            email = st.text_input("Email address *")
            phone = st.text_input("Phone number *", value="+91")
            address = st.text_input("Address / city")
            notes = st.text_area("Relevant history / symptoms")
            authorization_confirmed = st.checkbox(
                "I am authorized to create this patient record and assign the patient to my care."
            )
            submitted = st.form_submit_button("Create account & send credentials", type="primary")
        if submitted:
            full_name = f"{first.strip()} {last.strip()}".strip()
            if not first.strip() or not last.strip() or not email.strip() or not phone.strip():
                st.error("First name, last name, email and phone are required.")
            elif not validate_email(email) or not validate_phone(phone):
                st.error("Enter a valid email address and phone number with country code.")
            elif not authorization_confirmed:
                st.error("Confirm that you are authorized to create this patient record.")
            else:
                dob_value = next(iter(dob), None) if isinstance(dob, tuple) else dob
                ok, message, patient_id, temporary_password = create_doctor_added_patient_account(
                    doctor_id=doctor_id,
                    full_name=full_name,
                    email=email,
                    phone=phone,
                    dob=dob_value.strftime("%Y-%m-%d") if dob_value else None,
                    gender=gender,
                    address=address,
                    medical_notes=notes,
                )
                if ok and patient_id and temporary_password:
                    sent, delivery_message = send_patient_account_credentials(
                        email=email,
                        patient_name=full_name,
                        username=full_name,
                        temporary_password=temporary_password,
                        doctor_name=doctor.get("username", "Doctor"),
                    )
                    if sent:
                        st.success(message)
                        st.info("Credentials were delivered through the configured email provider.")
                        if not (os.getenv("SMTP_USERNAME") and os.getenv("SMTP_PASSWORD")):
                            st.warning(
                                "Development email mode: the temporary password was logged by the app server. "
                                "Configure SMTP before real use."
                            )
                    else:
                        st.error(
                            f"Patient account created and scan access enabled, but credentials were not sent: "
                            f"{delivery_message}"
                        )
                else:
                    st.error(message)
    pending = get_doctor_pending_invitations(doctor_id)
    if pending:
        with st.container(border=True):
            st.subheader("Invitations awaiting patient acceptance")
            st.dataframe(
                pd.DataFrame(pending)[["patient_name", "patient_email", "relationship_role", "created_at", "expires_at"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption("No patient scan, report or clinical history is available through these relationships until the patient accepts.")


def page_accept_invitation() -> None:
    token = st.session_state.get("patient_invitation_token", "")
    invitation = get_patient_invitation(token) if token else None
    st.markdown('<div class="doctor-workspace-marker"></div>', unsafe_allow_html=True)
    st.title("Patient relationship invitation")
    if not invitation:
        st.error("This invitation is invalid, expired or unavailable.")
        return
    if invitation["status"] != "pending":
        st.info(f"This invitation is {invitation['status']}.")
        return
    st.markdown('<div class="doctor-panel">', unsafe_allow_html=True)
    st.subheader(f"Invitation from Dr. {invitation['doctor_name']}")
    st.write(f"**Specialty:** {invitation.get('doctor_specialization') or 'Not specified'}")
    st.write(f"**Institution:** {invitation.get('institution') or invitation.get('doctor_hospital') or 'Not specified'}")
    st.write(f"**Relationship:** {invitation['relationship_role']}")
    if invitation.get("consent_note"):
        st.write(f"**Message:** {invitation['consent_note']}")
    st.warning("Your scans and reports remain private unless you accept this request. You can decline at any time.")
    st.markdown("</div>", unsafe_allow_html=True)

    patient_id = invitation["patient_id"]
    if is_logged_in() and current_user_id() != patient_id:
        st.error("Sign in to the patient account this invitation was sent to.")
        return

    if is_logged_in():
        with st.form("accept_existing_patient_invitation_form"):
            consent = st.checkbox(
                "I consent to sharing my NeuroScan records with this clinician and care team."
            )
            accepted = st.form_submit_button("Accept relationship", type="primary")
        if accepted:
            if not consent:
                st.error("Consent is required before this relationship can become active.")
            else:
                ok, message = respond_to_patient_invitation(token, patient_id, True)
                if ok:
                    st.success(message)
                    st.session_state["page"] = "dashboard"
                    st.rerun()
                else:
                    st.error(message)
    elif invitation["patient_phone_verified"]:
        with st.form("patient_invitation_login_form"):
            password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Sign in to review invitation")
        if login_submitted:
            logged_in, message, user = login_user(invitation["patient_email"], password)
            if logged_in and user and user["id"] == patient_id:
                set_session(user)
                st.rerun()
            else:
                st.error(message if not logged_in else "This invitation belongs to another account.")
    else:
        with st.form("new_patient_invitation_setup_form"):
            new_password = st.text_input("Create password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            consent = st.checkbox(
                "I consent to sharing my NeuroScan records with this clinician and care team."
            )
            setup_submitted = st.form_submit_button("Create password and accept", type="primary")
        if setup_submitted:
            password_ok, password_message = validate_password(new_password)
            if not password_ok:
                st.error(password_message)
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif not consent:
                st.error("Consent is required before the relationship can become active.")
            elif not update_user_password(patient_id, hash_password(new_password)):
                st.error("Could not save your password. Please try again.")
            else:
                ok, message = respond_to_patient_invitation(token, patient_id, True)
                if ok:
                    user = get_user_by_id(patient_id)
                    if user:
                        set_session(user)
                    st.success(message)
                    st.session_state["page"] = "dashboard"
                    st.rerun()
                else:
                    st.error(message)

    if st.button("Decline invitation"):
        ok, message = respond_to_patient_invitation(token, patient_id, False)
        if ok:
            st.info(message)
            st.session_state.pop("patient_invitation_token", None)
        else:
            st.error(message)


def page_doctor_scan_patient() -> None:
    doctor_id = _doctor_page_access()
    if not doctor_id:
        return
    _doctor_workspace_header(
        "Scan Patient",
        "Review the scan. Add the clinical context.",
        "Select an assigned patient, inspect the analysis and document your independent assessment.",
    )
    model_available = is_model_available()
    if not model_available:
        st.warning("MRI predictions are disabled because the trained model file is not available.")
    patients = get_doctor_assigned_patients_with_details(doctor_id)
    if not patients:
        with st.container(border=True):
            st.subheader("Patient selection")
            st.info(
                "No assigned patient is available. Add a patient account first to enable MRI access."
            )
            if st.button("Invite a patient", type="primary", key="scan_invite_patient"):
                st.session_state["page"] = "doctor_add_patient"
                st.rerun()
        return

    by_label = {f"{p['full_name']} · {p['patient_code']}": p for p in patients}
    labels = list(by_label)
    current_id = st.session_state.get("scanning_patient_id")
    current_idx = next((idx for idx, label in enumerate(labels) if by_label[label]["id"] == current_id), 0)
    with st.container(border=True):
        selected_label = st.selectbox(
            "Assigned patient",
            labels,
            index=current_idx,
            key="doctor_scan_patient_select",
        )
    target_patient = by_label.get(selected_label) if selected_label else None
    if not target_patient:
        st.info("Select an assigned patient to continue.")
        return
    patient_id = target_patient["id"]
    st.session_state["scanning_patient_id"] = patient_id
    allowed, auth_message = can_doctor_scan_patient(doctor_id, patient_id)
    if not allowed:
        with st.container(border=True):
            st.error(auth_message)
        return
    patient_code = target_patient["patient_code"]
    with st.container(border=True):
        st.write(f"**Patient:** {target_patient['full_name']}  ·  **Code:** `{patient_code}`")
        st.caption("Upload a JPG or PNG MRI image (up to 10 MB). AI output requires independent clinical review.")
        uploaded_file = st.file_uploader(
            "Upload MRI image (JPG or PNG)",
            type=["png", "jpg", "jpeg"],
            key=f"doctor_scan_uploader_{patient_id}",
        )
        if uploaded_file is not None and st.button(
            "Analyze scan and save to record",
            key=f"doctor_run_scan_{patient_id}",
            type="primary",
            disabled=not model_available,
        ):
            with st.spinner("Analyzing MRI..."):
                result = analyze_uploaded_scan(uploaded_file)
            prediction_id = _save_scan_result(result, uploaded_file, target_user_id=patient_id, doctor_id=doctor_id)
            log_audit_event(
                user_id=doctor_id,
                action="DOCTOR_PERFORMED_PATIENT_SCAN",
                resource=f"patient:{patient_id}",
                details={"prediction_id": prediction_id, "patient_code": patient_code, "class": result["class"]},
            )
            st.session_state[f"last_doc_scan_res_{patient_id}"] = (result, prediction_id)
            st.rerun()
    result_key = f"last_doc_scan_res_{patient_id}"
    if result_key in st.session_state:
        doctor = get_user_by_id(doctor_id) or {}
        result, prediction_id = st.session_state[result_key]
        _render_doctor_scan_review(
            result=result,
            prediction_id=prediction_id,
            target_patient=get_user_by_id(patient_id) or {"username": target_patient["full_name"]},
            patient_code=patient_code,
            doctor_id=doctor_id,
            doctor_name=f"Dr. {doctor.get('username', 'Doctor')}",
        )


def page_doctor_reports() -> None:
    doctor_id = _doctor_page_access()
    if not doctor_id:
        return
    _doctor_workspace_header(
        "Reports",
        "Review, document and export",
        "Patient reports combine model output with saved clinical review.",
    )
    reports = get_doctor_assigned_predictions(doctor_id)
    if not reports:
        with st.container(border=True):
            st.subheader("Report queue")
            st.info(
                "No patient scan reports are available yet. Reports appear here after an assigned patient has an MRI scan."
            )
            if st.button(
                "Invite a patient",
                type="primary",
                key="reports_invite_patient",
            ):
                st.session_state["page"] = "doctor_add_patient"
                st.rerun()
        return
    notes_by_report = {
        report["id"]: get_clinical_notes_for_prediction(report["id"])
        for report in reports
    }
    summary = st.columns(4)
    summary[0].metric("Scan reports", len(reports))
    summary[1].metric(
        "Awaiting review",
        sum(not notes for notes in notes_by_report.values()),
    )
    summary[2].metric(
        "Clinical notes saved",
        sum(bool(notes) for notes in notes_by_report.values()),
    )
    summary[3].metric(
        "Patients",
        len({report["user_id"] for report in reports}),
    )

    patient_labels = ["All patients"] + sorted({p["patient_username"] for p in reports})
    remembered_patient = st.session_state.get("report_patient_filter", "All patients")
    patient_index = patient_labels.index(remembered_patient) if remembered_patient in patient_labels else 0
    filters = st.columns([1.2, 1, 1.8])
    selected = filters[0].selectbox(
        "Patient",
        patient_labels,
        index=patient_index,
        key="doctor_report_patient_filter",
    )
    review_filter = filters[1].selectbox(
        "Review status",
        ["All reports", "Awaiting review", "Clinical notes saved"],
        key="doctor_report_review_filter",
    )
    search = filters[2].text_input(
        "Search reports",
        placeholder="Patient, report ID or MRI file",
        key="doctor_report_search",
    ).strip().lower()
    filtered = reports if selected == "All patients" else [p for p in reports if p["patient_username"] == selected]
    if review_filter != "All reports":
        filtered = [
            report for report in filtered
            if (
                review_filter == "Awaiting review"
                and not notes_by_report[report["id"]]
            ) or (
                review_filter == "Clinical notes saved"
                and bool(notes_by_report[report["id"]])
            )
        ]
    if search:
        filtered = [
            report for report in filtered
            if search in " ".join(
                str(report.get(field, ""))
                for field in ("patient_username", "id", "image_filename")
            ).lower()
        ]
    if not filtered:
        st.info("No reports match these filters.")
        return
    report_df = pd.DataFrame(filtered).rename(columns={
        "patient_username": "Patient",
        "image_filename": "MRI file",
        "prediction": "AI finding",
        "confidence": "Confidence",
        "severity": "Severity",
        "quality_label": "Image quality",
        "created_at": "Date",
    })
    report_df["Review status"] = [
        "Clinical notes saved" if notes_by_report[report["id"]] else "Awaiting review"
        for report in filtered
    ]
    if "Confidence" in report_df:
        report_df["Confidence"] = (
            pd.to_numeric(report_df["Confidence"], errors="raise")
            .mul(100)
            .map(lambda value: f"{value:.1f}%")
        )
    display_columns = [
        c for c in (
            "Patient", "MRI file", "AI finding", "Confidence",
            "Severity", "Image quality", "Review status", "Date"
        ) if c in report_df
    ]
    st.dataframe(report_df[display_columns], use_container_width=True, hide_index=True)
    csv_bytes = report_df[display_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered reports (CSV)",
        data=csv_bytes,
        file_name="neuroscan_patient_reports.csv",
        mime="text/csv",
    )
    report_options = {
        f"#{report['id']} · {report['patient_username']} · {report['created_at']}": report
        for report in filtered
    }
    selected_report = st.selectbox("Open report details", list(report_options))
    report_summary = report_options.get(selected_report) if selected_report else None
    if not report_summary:
        st.info("Select a report to view details.")
        return
    record = get_prediction_by_id(report_summary["id"])
    if not record or record["user_id"] != report_summary["user_id"]:
        st.error("This report is no longer available.")
        return
    with st.container(border=True):
        st.subheader(f"{report_summary['patient_username']} · MRI report #{record['id']}")
        detail1, detail2, detail3 = st.columns(3)
        detail1.metric("AI finding", record["prediction"])
        detail2.metric("Model confidence", f"{float(record['confidence']) * 100:.1f}%")
        detail3.metric("Image quality", record.get("quality_label") or "Not recorded")
        if record.get("ai_summary"):
            st.write(record["ai_summary"])
        notes = notes_by_report[record["id"]]
        if notes:
            st.markdown("**Clinical notes**")
            for note in notes:
                st.markdown(
                    f"- **{note['doctor_name']}** · {note['created_at']}: "
                    f"{note.get('recommendation') or 'No recommendation recorded.'}  \n"
                    f"  {note['note_content']}"
                )
        else:
            st.caption("No clinical notes have been saved for this scan.")

        with st.expander("Add independent clinical review"):
            with st.form(f"doctor_report_note_form_{record['id']}"):
                note_text = st.text_area(
                    "Clinical observation",
                    placeholder="Document your independent review and clinical context.",
                )
                recommendation = st.text_area(
                    "Clinical recommendation",
                    placeholder="Record follow-up or other care-plan recommendation.",
                )
                acknowledged = st.checkbox(
                    "I independently reviewed the scan and understand the AI output is not a diagnosis."
                )
                save_report_note = st.form_submit_button("Save clinical note", type="primary")
            if save_report_note:
                if not note_text.strip():
                    st.error("Enter a clinical observation before saving.")
                elif not acknowledged:
                    st.error("Confirm the clinical review before saving this note.")
                else:
                    add_clinical_note(
                        record["id"],
                        doctor_id,
                        note_text.strip(),
                        recommendation.strip() or None,
                    )
                    st.success("Clinical note saved to the patient record.")
                    st.rerun()

        image_bytes = record.get("image_data")
        if not image_bytes:
            st.warning("The stored scan image is unavailable, so a PDF with MRI visuals cannot be generated.")
        else:
            try:
                original_image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
            except (UnidentifiedImageError, OSError):
                st.warning("The stored image is not readable, so a PDF with MRI visuals cannot be generated.")
            else:
                steps = preprocess_array(original_image)
                probabilities = {
                    "Glioma": float(record.get("glioma_prob") or 0),
                    "Meningioma": float(record.get("meningioma_prob") or 0),
                    "Pituitary": float(record.get("pituitary_prob") or 0),
                    "Normal": float(record.get("normal_prob") or 0),
                }
                report_ai = _report_ai_fields(
                    record["prediction"],
                    float(record["confidence"]),
                    probabilities,
                    record.get("ai_summary"),
                )
                pdf_bytes = generate_pdf_report(
                    username=report_summary["patient_username"],
                    scan_date=str(record.get("created_at") or ""),
                    original_img=steps["original"],
                    preprocessed_img=steps["resized"],
                    gradcam_img=None,
                    prediction=record["prediction"],
                    confidence=float(record["confidence"]),
                    probabilities=probabilities,
                    class_info={**CLASS_INFO[record["prediction"]], "severity": record.get("severity")},
                    recommendations=[
                        note["recommendation"]
                        for note in notes
                        if note.get("recommendation")
                    ] or ["No clinician recommendation has been recorded for this scan."],
                    clinical_notes=notes,
                    scan_id=record["id"],
                    ai_summary=report_ai["summary"],
                    ai_sources=report_ai["sources"],
                    research_status=report_ai["status"],
                    model_name=record.get("model_name"),
                    quality_label=record.get("quality_label"),
                )
                st.download_button(
                    "Download PDF report",
                    data=pdf_bytes,
                    file_name=f"NeuroScan_Report_{record['id']:06d}.pdf",
                    mime="application/pdf",
                    key=f"doctor_report_pdf_{record['id']}",
                )
    st.caption("Export contains only reports from your currently assigned patients. AI findings require clinical interpretation.")


def page_patient_doctor_consultation() -> None:
    if not is_patient():
        st.error("Access Restricted: This page is for patient consultations.")
        return

    patient_id = current_user_id()
    _patient_workspace_header(
        "Consultation",
        "Make time for clarity",
        "Prepare for your next appointment and keep your care conversations together.",
    )
    assigned_doctors = get_assigned_doctors_for_patient(patient_id)
    appointments = get_patient_appointments(patient_id)
    scheduled_future = [
        appointment for appointment in appointments
        if appointment["status"] == "scheduled"
        and datetime.fromisoformat(appointment["scheduled_at"]) > datetime.now()
    ]
    next_appointment = min(
        scheduled_future,
        key=lambda appointment: appointment["scheduled_at"],
        default=None,
    )
    st.markdown(
        '<div style="text-align:right;margin:0 0 10px;">'
        '<a href="#schedule-consultation" style="display:inline-block;padding:8px 12px;'
        'border-radius:6px;background:#117c76;color:white;text-decoration:none;font-size:11px;'
        'font-weight:600;">＋ &nbsp; Schedule consultation</a></div>'
        '<div id="schedule-consultation"></div>',
        unsafe_allow_html=True,
    )
    appointment_col, schedule_col = st.columns([1.55, 1], gap="medium")
    with appointment_col:
        with st.container(border=True):
            st.markdown("#### Upcoming appointment")
            if next_appointment:
                scheduled = datetime.fromisoformat(next_appointment["scheduled_at"])
                st.markdown(f"### {scheduled.strftime('%d %b %Y')}")
                st.markdown(
                    f"**Dr. {html.escape(_doctor_display_name({'username': next_appointment['doctor_name']}))}** · "
                    f"{html.escape(next_appointment.get('specialization') or 'Specialist')}"
                )
                st.caption(
                    f"{scheduled.strftime('%I:%M %p')} · {next_appointment['appointment_type']} · "
                    f"{next_appointment.get('status', 'scheduled').title()}"
                )
                st.markdown(f"**Visit reason**  \n{html.escape(next_appointment['reason'])}")
                st.caption("Bring your complete MRI series, prior reports and questions.")
                with st.form(f"cancel_appointment_{next_appointment['id']}"):
                    confirm_cancel = st.checkbox(
                        "I understand this cancels the scheduled consultation.",
                        key=f"confirm_cancel_{next_appointment['id']}",
                    )
                    cancel_submitted = st.form_submit_button(
                        "Cancel appointment",
                        disabled=not confirm_cancel,
                    )
                if cancel_submitted:
                    if cancel_patient_appointment(patient_id, next_appointment["id"]):
                        st.success("Appointment cancelled.")
                        st.rerun()
                    else:
                        st.error("This appointment could not be cancelled. It may have already started or changed.")
            else:
                st.info("No upcoming consultation is scheduled.")
                st.caption("You can book a time with a doctor in your active care team.")
    with schedule_col:
        with st.container(border=True):
            st.markdown("#### Schedule a consultation")
            if not assigned_doctors:
                st.info("Connect with a verified doctor before scheduling.")
                if st.button("View My Doctors", use_container_width=True):
                    st.session_state["page"] = "patient_doctors"
                    st.rerun()
            else:
                doctor_by_label = {
                    f"Dr. {_doctor_display_name(doctor)} · {doctor.get('specialization') or 'Specialist'}": doctor
                    for doctor in assigned_doctors
                }
                default_doctor_index = next(
                    (
                        index for index, doctor in enumerate(doctor_by_label.values())
                        if doctor["id"] == st.session_state.get("consultation_doctor_id")
                    ),
                    0,
                )
                with st.form("patient_schedule_consultation"):
                    selected_doctor = st.selectbox(
                        "Doctor",
                        list(doctor_by_label),
                        index=default_doctor_index,
                    )
                    selected_date = st.date_input(
                        "Date",
                        value=date.today() + timedelta(days=1),
                        min_value=date.today(),
                    )
                    slots = [
                        time(hour, minute)
                        for hour in range(9, 17)
                        for minute in (0, 30)
                    ]
                    selected_time = st.selectbox(
                        "Available time",
                        slots,
                        format_func=lambda slot: slot.strftime("%I:%M %p"),
                    )
                    appointment_type = st.selectbox(
                        "Appointment type",
                        ["Video consultation", "In-person consultation"],
                    )
                    reason = st.text_area(
                        "Reason for visit",
                        placeholder="What would you like to discuss?",
                        max_chars=1000,
                    )
                    schedule_submitted = st.form_submit_button(
                        "Review and schedule",
                        type="primary",
                        use_container_width=True,
                    )
                if schedule_submitted:
                    doctor = doctor_by_label[selected_doctor]
                    scheduled_at = datetime.combine(selected_date, selected_time).isoformat(timespec="minutes")
                    created, message, _ = create_patient_appointment(
                        patient_id=patient_id,
                        doctor_id=doctor["id"],
                        scheduled_at=scheduled_at,
                        appointment_type=appointment_type,
                        reason=reason,
                    )
                    if created:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                st.caption("Times are shown in your local time. Each consultation is 30 minutes.")

    with st.container(border=True):
        st.markdown("#### Appointment history")
        if appointments:
            history = [
                {
                    "Date / time": datetime.fromisoformat(item["scheduled_at"]).strftime("%d %b %Y · %I:%M %p"),
                    "Doctor": f"Dr. {item['doctor_name']}",
                    "Purpose": item["reason"],
                    "Type": item["appointment_type"],
                    "Status": item["status"].title(),
                }
                for item in appointments
            ]
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        else:
            st.info("Your completed and cancelled consultations will appear here.")

    user_preds = get_user_predictions(patient_id)
    with st.container(border=True):
        st.markdown("#### Clinical feedback from your doctors")
        clinical_notes = [
            (scan, note)
            for scan in user_preds
            for note in get_clinical_notes_for_prediction(scan["id"])
        ]
        if clinical_notes:
            for scan, note in clinical_notes[:5]:
                st.markdown(
                    f"**Dr. {html.escape(_doctor_display_name({'username': note['doctor_name']}))}** "
                    f"· NS-{scan['id']:06d}"
                )
                st.caption(note.get("created_at") or "Date unavailable")
                if note.get("recommendation"):
                    st.markdown(f"**Recommendation:** {note['recommendation']}")
                st.write(note.get("note_content") or "")
        else:
            st.info("Signed clinician notes attached to your scans will appear here.")


def page_admin_dashboard() -> None:
    if not is_admin():
        st.error("Access Restricted: Administrator privileges required.")
        return

    _hero("⚙️ Admin Management Console", "Doctor verification, credential document inspection, and platform control.")
    admin_id = current_user_id()

    tab1, tab2, tab3 = st.tabs(["🩺 Doctor Verifications", "👥 User Management", "📊 System Metrics"])

    with tab1:
        _card_start()
        st.subheader("Doctor Qualification Verification Portal")
        pending_docs = get_all_doctors()
        pending_list = [d for d in pending_docs if d.get("status") in ("pending", "additional_info_required", "rejected", "suspended")]

        if not pending_list:
            st.success("No doctor verification applications pending review!")
        else:
            for doc in pending_list:
                status_color = "#fcbf49" if doc["status"] == "pending" else ("#ff6b6b" if doc["status"] in ("rejected", "suspended") else "#16b9ff")
                st.markdown(f"### Dr. {doc['username']} — Status: <span style='color:{status_color}; font-weight:bold;'>{doc['status'].upper()}</span>", unsafe_allow_html=True)

                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.write(f"**Email**: {doc['email']}")
                    st.write(f"**Phone**: {doc['phone']}")
                    st.write(f"**Specialization**: {doc.get('specialization', 'N/A')}")
                    st.write(f"**Qualification**: {doc.get('qualification', 'N/A')}")
                with dcol2:
                    st.write(f"**License No**: {doc.get('license_no', 'N/A')}")
                    st.write(f"**Hospital**: {doc.get('hospital', 'N/A')}")
                    st.write(f"**Experience**: {doc.get('years_experience', 'N/A')} years")
                    st.write(f"**Address**: {doc.get('address', 'N/A')}")

                docs = get_doctor_documents(doc["id"])
                st.markdown("#### 📎 Uploaded Medical Credentials")
                if not docs:
                    st.warning("No credential document records found for this doctor.")
                else:
                    doc_cols = st.columns(len(docs))
                    for col, dfile in zip(doc_cols, docs):
                        with col:
                            st.caption(f"**{dfile['doc_type'].upper()}**: {dfile['original_filename']}")
                            fpath = Path(dfile['file_path'])
                            if fpath.exists():
                                file_data = fpath.read_bytes()
                                st.download_button(
                                    label=f"⬇️ Download {dfile['doc_type'].capitalize()}",
                                    data=file_data,
                                    file_name=dfile['original_filename'],
                                    mime=dfile['mime_type'],
                                    key=f"dl_{dfile['id']}",
                                )
                                if dfile['mime_type'].startswith("image/"):
                                    st.image(file_data, caption=dfile['original_filename'], use_column_width=True)
                            else:
                                st.error("File not found on disk.")

                vnotes = st.text_input(f"Verification / Administrative Notes for Dr. {doc['username']}", key=f"vnote_{doc['id']}")

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    if st.button(f"✅ Approve Dr. {doc['username']}", key=f"app_{doc['id']}"):
                        update_doctor_verification(doc['id'], "approved", vnotes or "Approved by admin", admin_id)
                        update_user_status(doc['id'], "active", admin_id)
                        log_audit_event(admin_id, "verification_approved", resource=f"user:{doc['id']}")
                        st.success(f"Dr. {doc['username']} approved and verified!")
                        st.rerun()
                with b2:
                    if st.button(f"❓ Request Additional Info", key=f"req_{doc['id']}"):
                        if not vnotes.strip():
                            st.error("Please enter admin notes explaining what additional info is required.")
                        else:
                            update_doctor_verification(doc['id'], "additional_info_required", vnotes, admin_id)
                            log_audit_event(admin_id, "additional_info_requested", resource=f"user:{doc['id']}", details={"notes": vnotes})
                            st.info(f"Requested additional info from Dr. {doc['username']}.")
                            st.rerun()
                with b3:
                    if st.button(f"❌ Reject Application", key=f"rej_{doc['id']}"):
                        update_doctor_verification(doc['id'], "rejected", vnotes or "Rejected by admin", admin_id)
                        log_audit_event(admin_id, "verification_rejected", resource=f"user:{doc['id']}")
                        st.error(f"Dr. {doc['username']} application rejected.")
                        st.rerun()
                with b4:
                    if st.button(f"⛔ Suspend Account", key=f"susp_{doc['id']}"):
                        update_doctor_verification(doc['id'], "suspended", vnotes or "Suspended by admin", admin_id)
                        log_audit_event(admin_id, "account_suspended", resource=f"user:{doc['id']}")
                        st.warning(f"Dr. {doc['username']} account suspended.")
                        st.rerun()
                st.markdown("---")
        _card_end()

    with tab2:
        _card_start()
        st.subheader("All Registered Users")
        users = get_all_users()
        if users:
            udf = pd.DataFrame(users)
            st.dataframe(udf[["id", "username", "email", "role", "status", "specialization", "created_at"]], use_container_width=True)

            st.markdown("---")
            st.subheader("Modify Account Status")
            user_options = {f"{u['username']} (ID: {u['id']} - Role: {u['role']} - Status: {u['status']})": u['id'] for u in users}
            selected_user_label = st.selectbox("Select User", list(user_options.keys()))
            selected_user_id = user_options[selected_user_label]
            new_status = st.selectbox("Set New Status", ["active", "inactive", "pending", "rejected"])

            if st.button("Update User Status"):
                update_user_status(selected_user_id, new_status, admin_id)
                st.success(f"Status for user ID {selected_user_id} updated to {new_status}.")
                st.rerun()
        _card_end()

    with tab3:
        _card_start()
        st.subheader("Platform Aggregate Metrics")
        users = get_all_users()
        total_users = len(users)
        total_docs = len([u for u in users if u["role"] == "doctor"])
        total_patients = len([u for u in users if u["role"] == "patient"])

        mcol1, mcol2, mcol3 = st.columns(3)
        _metric_tile(mcol1, str(total_users), "Total Registered Users", "#16b9ff")
        _metric_tile(mcol2, str(total_docs), "Medical Doctors", "#00d4a5")
        _metric_tile(mcol3, str(total_patients), "Patients", "#fcbf49")
        _card_end()


def _admin_workspace_header(eyebrow: str, title: str, subtitle: str, action: str | None = None) -> None:
    st.markdown('<div class="admin-workspace-marker"></div>', unsafe_allow_html=True)
    action_html = (
        f'<span class="admin-status">{html.escape(action)}</span>' if action else ""
    )
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px;">'
        f'<div><div class="admin-eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1 class="admin-shell-title">{html.escape(title)}</h1>'
        f'<div class="admin-shell-subtitle">{html.escape(subtitle)}</div></div>{action_html}</div>',
        unsafe_allow_html=True,
    )


def _admin_metrics(users: list[dict], doctors: list[dict]) -> None:
    patients = [u for u in users if u.get("role") == "patient"]
    verified = [d for d in doctors if d.get("status") in ("active", "approved")]
    pending = [d for d in doctors if d.get("status") in ("pending", "additional_info_required")]
    metrics = st.columns(3)
    metrics[0].metric("Total users", len(users), f"{len(patients)} patients · {len(doctors)} doctors")
    metrics[1].metric("Patients", len(patients), "Registered patient accounts")
    metrics[2].metric("Doctors", len(doctors), f"{len(verified)} verified · {len(pending)} pending")


def page_admin_dashboard() -> None:
    if not is_admin():
        st.error("Access Restricted: Administrator privileges required.")
        return
    _admin_workspace_header(
        "Admin Console / Overview",
        "Admin Console",
        "Account governance, professional verification and platform health.",
        "Secure session",
    )
    users = get_all_users()
    doctors = get_all_doctors()
    _admin_metrics(users, doctors)
    pending = [d for d in doctors if d.get("status") in ("pending", "additional_info_required")]
    left, right = st.columns([1.7, .8], gap="medium")
    with left:
        st.markdown('<div class="admin-panel"><h3>Doctor verification queue</h3>', unsafe_allow_html=True)
        if pending:
            pending_df = pd.DataFrame(pending).reindex(
                columns=["username", "specialization", "status"]
            )
            st.dataframe(
                pending_df.head(6),
                use_container_width=True, hide_index=True,
            )
            if st.button("Review doctors", type="primary"):
                st.session_state["page"] = "admin_doctor_verification"
                st.rerun()
        else:
            st.success("No doctor applications are awaiting review.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="admin-panel"><h3>Platform scan volume</h3><div class="admin-muted">Illustrative activity from recorded scan reports</div>', unsafe_allow_html=True)
        scan_counts = []
        for user in users:
            if user.get("role") == "patient":
                scan_counts.extend(get_user_predictions(user["id"]))
        if len(scan_counts) >= 2:
            chart = pd.DataFrame(
                {"Scans": range(1, len(scan_counts) + 1)},
                index=range(1, len(scan_counts) + 1),
            )
            st.plotly_chart(
                px.line(chart, x=chart.index, y="Scans", markers=True),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("Scan activity will appear here after at least two reports are created.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div class="admin-panel" style="background:rgba(17,124,118,.08);">'
            '<h3>System health</h3><p class="admin-status">All systems operational</p>'
            '<p class="admin-muted">Inference service · Available</p>'
            '<p class="admin-muted">Storage · Available</p>'
            '<p class="admin-muted">Audit pipeline · Available</p></div>'
            '<div class="admin-note">Governance-only workspace. Administrative tools do not include patient diagnosis or clinical care workflows.</div>',
            unsafe_allow_html=True,
        )


def page_admin_doctor_verification() -> None:
    if not is_admin():
        st.error("Access Restricted: Administrator privileges required.")
        return
    _admin_workspace_header(
        "Platform Oversight",
        "Verify expertise. Build trust.",
        "Review qualifications, confirm license details and record each credential decision.",
        "Export review queue",
    )
    admin_id = current_user_id()
    doctors = get_all_doctors()
    pending = [d for d in doctors if d.get("status") in ("pending", "additional_info_required", "rejected", "suspended")]
    st.columns(3)[0].metric("Credential reviews", len(pending), "Applications needing attention")
    st.columns(3)[1].metric("Awaiting information", len([d for d in pending if d.get("status") == "additional_info_required"]))
    st.columns(3)[2].metric("Oldest application", "—" if not pending else "Needs review")
    if not pending:
        st.success("No doctor verification applications are awaiting review.")
        return
    for doc in pending:
        with st.container(border=True):
            st.markdown(f"### Dr. {html.escape(doc['username'])}")
            st.caption(f"{doc.get('specialization') or 'Specialization not provided'} · {doc.get('status', '').replace('_', ' ').title()}")
            details = st.columns(2)
            details[0].write(f"**Qualification:** {doc.get('qualification') or 'Not provided'}")
            details[0].write(f"**License:** {doc.get('license_no') or 'Not provided'}")
            details[1].write(f"**Medical facility:** {doc.get('hospital') or 'Not provided'}")
            details[1].write(f"**Experience:** {doc.get('years_experience') or 0} years")
            docs = get_doctor_documents(doc["id"])
            if docs:
                st.caption("Supporting documents")
                for dfile in docs:
                    path = Path(dfile["file_path"])
                    if path.exists():
                        st.download_button(
                            f"View {dfile['original_filename']}", path.read_bytes(),
                            file_name=dfile["original_filename"], mime=dfile["mime_type"],
                            key=f"admin_doc_{dfile['id']}",
                        )
            note = st.text_area("Review note", key=f"admin_review_note_{doc['id']}", placeholder="Record the decision reason.")
            actions = st.columns(4)
            if actions[0].button("Approve", key=f"admin_approve_{doc['id']}", type="primary"):
                update_doctor_verification(doc["id"], "approved", note or "Approved by admin", admin_id)
                update_user_status(doc["id"], "active", admin_id)
                log_audit_event(admin_id, "verification_approved", resource=f"user:{doc['id']}")
                st.rerun()
            if actions[1].button("Reject", key=f"admin_reject_{doc['id']}"):
                update_doctor_verification(doc["id"], "rejected", note or "Rejected by admin", admin_id)
                log_audit_event(admin_id, "verification_rejected", resource=f"user:{doc['id']}")
                st.rerun()
            if actions[2].button("Request information", key=f"admin_info_{doc['id']}"):
                if not note.strip():
                    st.error("Add a review note before requesting information.")
                else:
                    update_doctor_verification(doc["id"], "additional_info_required", note, admin_id)
                    log_audit_event(admin_id, "additional_info_requested", resource=f"user:{doc['id']}")
                    st.rerun()
            if actions[3].button("Suspend", key=f"admin_suspend_{doc['id']}"):
                update_doctor_verification(doc["id"], "suspended", note or "Suspended by admin", admin_id)
                update_user_status(doc["id"], "inactive", admin_id)
                st.rerun()


def page_admin_user_management() -> None:
    if not is_admin():
        st.error("Access Restricted: Administrator privileges required.")
        return
    _admin_workspace_header("Platform Oversight", "Manage access with care", "Search role-based accounts, review status and apply controlled access changes.", "Invite user")
    users = get_all_users()
    doctors = [u for u in users if u.get("role") == "doctor"]
    pending = [u for u in doctors if u.get("status") in ("pending", "additional_info_required")]
    _admin_metrics(users, doctors)
    st.markdown('<div class="admin-panel"><h3>Account directory</h3><div class="admin-muted">Illustrative accounts and clinical records are not shown in administration.</div>', unsafe_allow_html=True)
    search = st.text_input("Search name, email or user ID", placeholder="Search account directory")
    filtered = [u for u in users if not search.strip() or search.lower() in f"{u.get('username','')} {u.get('email','')} {u.get('id','')}".lower()]
    if filtered:
        st.dataframe(pd.DataFrame(filtered)[["id", "username", "email", "role", "status", "created_at"]], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if not filtered:
        return
    selected = st.selectbox("Selected account", [f"{u['username']} · {u['email']}" for u in filtered])
    selected_user = filtered[[f"{u['username']} · {u['email']}" for u in filtered].index(selected)]
    st.markdown('<div class="admin-panel"><h3>Access change confirmation</h3>', unsafe_allow_html=True)
    new_status = st.selectbox("Account status", ["active", "inactive", "pending", "rejected"])
    if st.button("Save access change", type="primary"):
        update_user_status(selected_user["id"], new_status, current_user_id())
        log_audit_event(current_user_id(), "admin_user_status_updated", resource=f"user:{selected_user['id']}", details={"status": new_status})
        st.success("Account status updated.")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def page_admin_system_metrics() -> None:
    if not is_admin():
        st.error("Access Restricted: Administrator privileges required.")
        return
    _admin_workspace_header("Platform Oversight", "A clear view of platform health", "Monitor service operation and illustrative MRI analytics, independently of clinical performance.", "Export telemetry")
    users = get_all_users()
    doctors = get_all_doctors()
    scans = []
    for user in users:
        if user.get("role") == "patient":
            scans.extend(get_user_predictions(user["id"]))
    metrics = st.columns(4)
    metrics[0].metric("Registered users", len(users))
    metrics[1].metric("MRI analyses", len(scans))
    metrics[2].metric("Platform availability", "99.9%")
    metrics[3].metric("Average analysis time", "—")
    left, right = st.columns([1.5, .8])
    with left:
        st.markdown('<div class="admin-panel"><h3>MRI scan volume</h3>', unsafe_allow_html=True)
        if len(scans) >= 2:
            scan_chart = pd.DataFrame(
                {
                    "Analysis": range(1, len(scans) + 1),
                    "Completed MRI analyses": range(1, len(scans) + 1),
                }
            )
            st.plotly_chart(
                px.bar(scan_chart, x="Analysis", y="Completed MRI analyses"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("MRI volume charts will appear after at least two analyses are recorded.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="admin-panel"><h3>Service health</h3>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"Service": "MRI analysis", "State": "Operational", "Last checked": "Now"},
            {"Service": "Report generation", "State": "Operational", "Last checked": "Now"},
            {"Service": "Account authentication", "State": "Operational", "Last checked": "Now"},
            {"Service": "Notification delivery", "State": "Configured" if os.getenv("SMTP_USERNAME") else "Not configured", "Last checked": "Now"},
        ]), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="admin-panel"><h3>Tumour-class analytics</h3><div class="admin-muted">AI prediction distribution, not a measure of disease prevalence.</div>', unsafe_allow_html=True)
        classes = {}
        for scan in scans:
            classes[scan.get("prediction", "Unknown")] = classes.get(scan.get("prediction", "Unknown"), 0) + 1
        if classes:
            class_chart = pd.DataFrame(
                {"Class": list(classes), "Predictions": list(classes.values())}
            )
            st.plotly_chart(
                px.bar(class_chart, x="Class", y="Predictions"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("Tumour-class analytics will appear after the first saved analysis.")
        st.markdown("</div>", unsafe_allow_html=True)


def page_audit_logs() -> None:
    if not is_admin():
        st.error("Access Restricted: Administrator privileges required.")
        return

    _admin_workspace_header(
        "Workspace / Audit Logs",
        "Every action, traceable",
        "Review role-based activity, decision reasons and account-access events.",
        "Export filtered logs",
    )
    _card_start()
    logs = get_audit_logs(limit=150)
    if not logs:
        st.info("No audit logs recorded yet.")
    else:
        log_df = pd.DataFrame(logs)
        st.dataframe(
            log_df[["id", "timestamp", "user_id", "username", "role", "action", "resource", "details_json"]],
            use_container_width=True,
        )
    _card_end()


def page_profile() -> None:
    if not is_logged_in():
        st.error("Access Restricted: Please log in to view your profile.")
        return

    user_id = current_user_id()
    if user_id is None:
        st.error("User session is unavailable. Please sign in again.")
        return
    user = get_user_by_id(user_id)
    if not user:
        st.error("User record not found.")
        return
    if user.get("role") == "admin":
        _admin_workspace_header(
            "Workspace / Profile",
            "Your administrator profile",
            "Manage identity, multi-factor authentication and operational notification preferences.",
            "Administrator · MFA required",
        )
        left, right = st.columns([.8, 2], gap="medium")
        with left:
            st.markdown(
                f'<div class="admin-panel"><h3>{html.escape(user.get("username", "Administrator"))}</h3>'
                '<p class="admin-muted">AD-0001 · Platform administrator</p>'
                '<p class="admin-status">Active account</p><hr>'
                '<p class="admin-muted">Account details</p><p class="admin-muted">Access & identity</p>'
                '<p class="admin-muted">Privacy & preferences</p><p class="admin-muted">Security</p></div>'
                '<div class="admin-note">Keep your contact details and security settings current.</div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown('<div class="admin-panel"><h3>Personal details</h3><div class="admin-muted">Review your identity and contact information</div>', unsafe_allow_html=True)
            details = st.columns(2)
            details[0].text_input("Full name", value=user.get("username") or "", disabled=True)
            details[1].text_input("Email address", value=user.get("email") or "", disabled=True)
            details[0].text_input("Phone number", value=user.get("phone") or "", disabled=True)
            details[1].text_input("Time zone", value="UTC · Coordinated Universal Time", disabled=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="admin-panel"><h3>Access & identity</h3><div class="admin-note">Administrative access does not permit clinical sign-off. Role changes require an independent administrator review.</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="admin-panel"><h3>Preferences & privacy</h3>', unsafe_allow_html=True)
            st.toggle("Credential review alerts", value=True)
            st.toggle("Platform health notifications", value=True)
            st.toggle("Optional product updates", value=False)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="admin-panel"><h3>Security</h3><div class="admin-muted">Manage authentication and current sessions.</div>', unsafe_allow_html=True)
            st.write("**Password** · Last changed recently")
            st.button("Change password", key="admin_change_password")
            st.toggle("Multi-factor authentication", value=True)
            st.markdown("</div>", unsafe_allow_html=True)
        return

    role = user.get("role", "patient").capitalize()
    status = user.get("status", "active").upper()
    is_verified_doctor = user.get("role") == "doctor" and status.lower() in ("active", "approved")
    if is_verified_doctor:
        _doctor_workspace_header(
            "My Profile",
            f"Dr. {_doctor_display_name(user)}",
            "Review your professional details, credentials and account security.",
        )
        docs = get_doctor_documents(user_id)
        assigned_patients = get_doctor_assigned_patients_with_details(user_id)
        metrics = st.columns(4)
        metrics[0].metric("Verification", status.title())
        metrics[1].metric("Phone verification", "Verified" if user.get("is_verified") else "Not verified")
        metrics[2].metric("My patients", len(assigned_patients))
        metrics[3].metric("Credential files", len(docs))

        details_tab, security_tab = st.tabs(["Professional profile", "Account security"])
        with details_tab:
            left, right = st.columns([1.25, 1], gap="large")
            with left:
                with st.container(border=True):
                    st.subheader("Professional information")
                    st.write(f"**Specialty:** {user.get('specialization') or 'Not provided'}")
                    st.write(f"**Qualification:** {user.get('qualification') or 'Not provided'}")
                    st.write(f"**Medical registration:** {user.get('license_no') or 'Not provided'}")
                    st.caption("Verified qualifications and registration details are read-only. Contact an administrator to correct them.")
                    with st.form("doctor_profile_contact_form"):
                        hospital = st.text_input("Hospital / clinic", value=user.get("hospital") or "")
                        address = st.text_area("Practice address", value=user.get("address") or "")
                        years_experience = st.number_input(
                            "Years of experience",
                            min_value=0,
                            max_value=80,
                            value=max(0, min(80, int(user.get("years_experience") or 0))),
                        )
                        save_profile = st.form_submit_button("Save profile changes", type="primary")
                if save_profile:
                    if update_doctor_profile(user_id, hospital, address, int(years_experience)):
                        st.success("Professional profile updated.")
                        st.rerun()
                    else:
                        st.error("Could not save your profile. Please retry.")
            with right:
                with st.container(border=True):
                    st.subheader("Contact & account")
                    st.write(f"**Name:** Dr. {_doctor_display_name(user)}")
                    st.write(f"**Email:** {user['email']}")
                    st.write(f"**Phone:** {user['phone']}")
                    st.write(f"**Account created:** {user.get('created_at') or 'Not available'}")
                    st.write(f"**Last sign-in:** {user.get('last_login') or 'Not available'}")

                with st.container(border=True):
                    st.subheader("Credential documents")
                    if not docs:
                        st.info("No credential files are attached to this account.")
                    for dfile in docs:
                        file_path = Path(dfile["file_path"])
                        st.write(f"**{dfile['doc_type'].replace('_', ' ').title()}** · {dfile['original_filename']}")
                        if file_path.exists():
                            st.download_button(
                                "Download document",
                                data=file_path.read_bytes(),
                                file_name=dfile["original_filename"],
                                mime=dfile["mime_type"],
                                key=f"prof_dl_{dfile['id']}",
                            )
                        else:
                            st.error(f"Credential file is missing: {dfile['original_filename']}")
        with security_tab:
            with st.container(border=True):
                st.subheader("Change password")
                with st.form("doctor_change_password_form"):
                    current_password = st.text_input("Current password", type="password")
                    new_password = st.text_input("New password", type="password")
                    confirm_password = st.text_input("Confirm new password", type="password")
                    change_password = st.form_submit_button("Update password", type="primary")
            if change_password:
                if not verify_password(current_password, user.get("password", "")):
                    st.error("Current password is incorrect.")
                else:
                    password_ok, password_message = validate_password(new_password)
                    if not password_ok:
                        st.error(password_message)
                    elif new_password != confirm_password:
                        st.error("The new passwords do not match.")
                    elif not update_user_password(user_id, hash_password(new_password)):
                        st.error("Password could not be updated.")
                    else:
                        log_audit_event(
                            user_id=user_id,
                            action="doctor_password_changed",
                            resource=f"user:{user_id}",
                        )
                        st.success("Password updated successfully.")
            st.caption("If you cannot sign in, use the password recovery flow on the login page.")
    elif is_patient():
        display_name = _patient_display_name(user)
        _patient_workspace_header(
            "Profile",
            "Your profile, your preferences",
            "Keep your personal details and medical context up to date.",
        )
        patient_profile = get_patient_profile(user_id) or {}
        scans = get_user_predictions(user_id, limit=200)
        doctors = get_assigned_doctors_for_patient(user_id)
        left, right = st.columns([0.72, 2.1], gap="medium")
        with left:
            with st.container(border=True):
                avatar = html.escape(display_name[:1].upper())
                st.markdown(
                    f'<div style="display:grid;place-items:center;width:42px;height:42px;'
                    f'border-radius:50%;background:#e5f5f3;color:#117c76;font-weight:700;">{avatar}</div>'
                    f'<h3>{html.escape(display_name)}</h3>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Patient · {user.get('status', 'active').title()} account")
                st.markdown("---")
                st.caption("Account details")
                st.caption("Medical context")
                st.caption("Privacy & care team")
                st.caption("Security")
                st.markdown("---")
                st.caption(f"{len(scans)} saved MRI scans")
                st.caption(f"{len(doctors)} active doctor relationships")
            with st.container(border=True):
                st.markdown("#### Your privacy")
                st.caption(
                    "Only verified doctors in active care relationships can access your shared scans. "
                    "You can review your care team at any time."
                )
                if st.button("Manage My Doctors", use_container_width=True):
                    st.session_state["page"] = "patient_doctors"
                    st.rerun()
        with right:
            with st.container(border=True):
                st.markdown("#### Personal details")
                st.caption("Your account name and contact details are shown as registered.")
                details_left, details_right = st.columns(2)
                details_left.text_input("Full name", value=display_name, disabled=True)
                details_right.text_input("Email address", value=user.get("email") or "", disabled=True)
                details_left.text_input("Phone number", value=user.get("phone") or "", disabled=True)
                details_right.text_input(
                    "Account created",
                    value=str(user.get("created_at") or "Not available"),
                    disabled=True,
                )
            with st.container(border=True):
                st.markdown("#### Medical context")
                st.caption("Optional context for your care team; it does not change model output.")
                gender_options = ["Not provided", "Female", "Male", "Other"]
                profile_gender = patient_profile.get("patient_gender") or "Not provided"
                gender_index = (
                    gender_options.index(profile_gender)
                    if profile_gender in gender_options
                    else 0
                )
                saved_dob = None
                if patient_profile.get("dob"):
                    try:
                        saved_dob = date.fromisoformat(patient_profile["dob"])
                    except ValueError:
                        st.warning("The saved date of birth is not a valid date. Select a replacement if needed.")
                    if saved_dob and saved_dob > date.today():
                        st.warning("The saved date of birth is in the future. Select a valid date if needed.")
                        saved_dob = None
                with st.form("patient_medical_profile_form"):
                    form_left, form_right = st.columns(2)
                    with form_left:
                        dob = st.date_input(
                            "Date of birth",
                            value=saved_dob,
                            max_value=date.today(),
                        )
                    with form_right:
                        gender = st.selectbox(
                            "Sex recorded for care",
                            gender_options,
                            index=gender_index,
                        )
                    address = st.text_input(
                        "Address",
                        value=patient_profile.get("address") or "",
                    )
                    medical_history = st.text_area(
                        "Relevant history / symptoms",
                        value=patient_profile.get("medical_history") or "",
                        placeholder="Optional context to discuss with your doctor.",
                        max_chars=3000,
                    )
                    save_patient_profile = st.form_submit_button(
                        "Save medical context",
                        type="primary",
                    )
                if save_patient_profile:
                    saved = update_patient_profile(
                        user_id,
                        None if gender == "Not provided" else gender,
                        dob.isoformat() if dob else None,
                        address,
                        medical_history,
                    )
                    if saved:
                        st.success("Medical context saved.")
                        st.rerun()
                    else:
                        st.error("Medical context could not be saved.")
            with st.container(border=True):
                st.markdown("#### Security")
                st.caption("Change your password. Never share it with your care team.")
                with st.form("patient_change_password_form"):
                    current_password = st.text_input("Current password", type="password")
                    new_password = st.text_input("New password", type="password")
                    confirm_password = st.text_input("Confirm new password", type="password")
                    change_password = st.form_submit_button("Update password", type="primary")
                if change_password:
                    if not verify_password(current_password, user.get("password", "")):
                        st.error("Current password is incorrect.")
                    else:
                        password_ok, password_message = validate_password(new_password)
                        if not password_ok:
                            st.error(password_message)
                        elif new_password != confirm_password:
                            st.error("The new passwords do not match.")
                        elif not update_user_password(user_id, hash_password(new_password)):
                            st.error("Password could not be updated.")
                        else:
                            log_audit_event(
                                user_id=user_id,
                                action="patient_password_changed",
                                resource=f"user:{user_id}",
                            )
                            st.success("Password updated successfully.")
    else:
        _hero(f"👤 {user['username']}'s Profile", f"Account details and security ({role})")
        metrics = st.columns(4)
        metrics[0].metric("Account role", role)
        metrics[1].metric("Account status", status)
        metrics[2].metric("Phone verification", "Verified" if user.get("is_verified") else "Not verified")
        metrics[3].metric("User ID", f"#{user['id']:05d}")
        st.markdown('<div class="doctor-panel">', unsafe_allow_html=True)
        st.subheader("Account details")
        st.write(f"**Email:** {user['email']}")
        st.write(f"**Phone:** {user['phone']}")
        st.write(f"**Account created:** {user.get('created_at') or 'Not available'}")
        st.markdown("</div>", unsafe_allow_html=True)
        if user.get("role") == "patient":
            st.markdown('<div class="doctor-panel">', unsafe_allow_html=True)
            st.subheader("Patient clinical summary")
            user_preds = get_user_predictions(user_id)
            assigned_docs = get_assigned_doctors_for_patient(user_id)
            summary_cols = st.columns(2)
            summary_cols[0].metric("MRI scans", len(user_preds))
            summary_cols[1].metric("Assigned specialists", len(assigned_docs))
            st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    auth_token = st.query_params.get("auth")
    restored_auth = False
    if auth_token and not is_logged_in():
        if restore_session_from_token(auth_token):
            restored_auth = True
        else:
            del st.query_params["auth"]

    query_page = st.query_params.get("page")
    if query_page == "accept_invitation":
        st.session_state["page"] = "accept_invitation"
        st.session_state["patient_invitation_token"] = st.query_params.get("token", "")
        del st.query_params["page"]
        if "token" in st.query_params:
            del st.query_params["token"]
    if query_page in {"home", "login", "register", "forgot_password"}:
        st.session_state["page"] = query_page
        del st.query_params["page"]

    if restored_auth and "page" not in st.session_state:
        if is_admin():
            st.session_state["page"] = "admin_dashboard"
        elif is_doctor():
            st.session_state["page"] = (
                "doctor_dashboard"
                if current_user_status() in ("active", "approved")
                else "doctor_pending"
            )
        elif is_patient():
            st.session_state["page"] = "dashboard"

    page = st.session_state.get("page", "home")
    # Render the workspace marker before the sidebar so its CSS selectors are
    # available during the same Streamlit render as the navigation controls.
    if page in {"doctor_pending", "doctor_pending_profile", "doctor_pending_mastishka"}:
        st.markdown('<div class="pending-workspace-marker"></div>', unsafe_allow_html=True)
    elif page == "accept_invitation":
        st.markdown('<div class="doctor-workspace-marker"></div>', unsafe_allow_html=True)
    elif is_admin():
        st.markdown('<div class="admin-workspace-marker"></div>', unsafe_allow_html=True)
    elif is_doctor() and current_user_status() in ("active", "approved"):
        st.markdown('<div class="doctor-workspace-marker"></div>', unsafe_allow_html=True)
    elif is_patient() and current_user_status() == "active":
        st.markdown('<div class="patient-workspace-marker"></div>', unsafe_allow_html=True)

    if page in {"doctor_pending", "doctor_pending_profile", "doctor_pending_mastishka"}:
        inject_doctor_workspace_theme()
    elif page == "accept_invitation" or (
        is_doctor() and current_user_status() in ("active", "approved")
    ):
        inject_doctor_workspace_theme()
    elif is_admin():
        inject_admin_workspace_theme()
    elif is_patient() and current_user_status() == "active":
        inject_patient_workspace_theme()
    if page != "accept_invitation":
        render_sidebar()
    routes = {
        "home": page_home,
        "register": page_register,
        "login": page_login,
        "forgot_password": page_forgot_password,
        "accept_invitation": page_accept_invitation,
        "upload": page_upload,
        "dashboard": page_dashboard,
        "history": page_history,
        "patient_reports": page_patient_reports,
        "patient_doctors": page_patient_doctors,
        "training": page_training,
        "mastishka": page_mastishka,
        "about": page_about,
        "profile": page_profile,
        "doctor_dashboard": page_doctor_dashboard,
        "doctor_patients": page_doctor_patients,
        "doctor_add_patient": page_doctor_add_patient,
        "doctor_scan_patient": page_doctor_scan_patient,
        "doctor_reports": page_doctor_reports,
        "doctor_pending": page_doctor_pending,
        "doctor_pending_profile": page_doctor_pending_profile,
        "doctor_pending_mastishka": page_doctor_pending_mastishka,
        "consultation": page_patient_doctor_consultation,
        "admin_dashboard": page_admin_dashboard,
        "admin_doctor_verification": page_admin_doctor_verification,
        "admin_user_management": page_admin_user_management,
        "admin_system_metrics": page_admin_system_metrics,
        "audit_logs": page_audit_logs,
    }
    routes.get(page, page_home)()


if __name__ == "__main__":
    main()
