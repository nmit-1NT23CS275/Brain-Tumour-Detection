"""Reusable Streamlit theme helpers."""

from __future__ import annotations

import streamlit as st


_THEME = """
<style>
@import url('https://fonts.cdnfonts.com/css/open-sauce-one');
:root {
  --ink:#193d45;
  --teal:#117c76;
  --mist:#f4f8f8;
  --paper:#ffffff;
  --line:rgba(25,61,69,.14);
  --muted:rgba(25,61,69,.74);
  --soft:rgba(25,61,69,.06);
}
html,body,[class*="css"] {
  font-family:'Open Sauce One',sans-serif !important;
  color:var(--ink);
}
body,[data-testid="stAppViewContainer"],[data-testid="stMain"] {
  background:var(--mist) !important;
  color:var(--ink) !important;
}
body [class*="css"] {font-family:'Open Sauce One',sans-serif !important;}
section.main > div.block-container {
  max-width:1560px;
  padding:30px 40px 56px;
}
header[data-testid="stHeader"],footer {background:transparent;}
.stDeployButton,[data-testid="stToolbar"],#MainMenu {display:none !important;}
[data-testid="stStatusWidget"],[data-testid="InputInstructions"] {display:none !important;}
h1,h2,h3,h4,h5,h6 {color:var(--ink) !important;letter-spacing:-.02em;}
h1 {font-size:34px !important;line-height:1.2 !important;font-weight:700 !important;}
h2 {font-size:25px !important;line-height:1.3 !important;font-weight:700 !important;}
h3 {font-size:20px !important;line-height:1.35 !important;font-weight:700 !important;}
h4,h5,h6 {font-size:17px !important;line-height:1.4 !important;font-weight:700 !important;}
p,li,label,[data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"] {
  color:var(--ink);
}
[data-testid="stMarkdownContainer"] p,[data-testid="stCaptionContainer"] p {
  font-size:15px;
  line-height:1.6;
}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p {
  color:var(--muted) !important;
}
[data-testid="stWidgetLabel"] p,[data-testid="stFileUploader"] label,
[data-testid="stCheckbox"] label,[data-testid="stRadio"] label {
  color:var(--ink) !important;font-size:15px !important;font-weight:600;
}
[data-testid="stMetric"],.metric-tile {
  color:var(--ink);background:var(--paper);border:1px solid var(--line);
  border-radius:16px;padding:18px 20px;box-shadow:0 8px 24px rgba(25,61,69,.06);
}
[data-testid="stMetricLabel"] p,.metric-label {
  color:var(--muted) !important;font-size:13px !important;
}
[data-testid="stMetricValue"],.metric-value {
  color:var(--ink) !important;font-size:28px !important;font-weight:700;
}
[data-testid="stForm"] {
  background:transparent;border:0;border-radius:0;box-shadow:none;padding:0;
}
.app-card,.doctor-panel,.patient-panel {
  background:var(--paper);color:var(--ink);border:1px solid var(--line);
  border-radius:16px;padding:22px;box-shadow:0 8px 24px rgba(25,61,69,.05);
  margin-bottom:18px;
}
.app-card p,.doctor-panel p,.patient-panel p {color:var(--ink);}
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stDateInput"] div[data-baseweb="input"],
[data-testid="stTimeInput"] div[data-baseweb="input"],
[data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stMultiSelect"] div[data-baseweb="select"] {
  min-height:46px;background:#161b22 !important;
  border:1px solid var(--line) !important;border-radius:10px !important;
  box-shadow:none !important;
}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input,[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input,[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stMultiSelect"] [role="combobox"] {
  min-height:46px;background:transparent !important;color:#fff !important;
  -webkit-text-fill-color:#fff !important;
  border:0 !important;border-radius:0 !important;box-shadow:none !important;
  outline:0 !important;font-family:'Open Sauce One',sans-serif !important;font-size:15px !important;
}
[data-testid="stTextInput"] input:disabled,
[data-testid="stNumberInput"] input:disabled,
[data-testid="stDateInput"] input:disabled,
[data-testid="stTimeInput"] input:disabled,
[data-testid="stTextArea"] textarea:disabled {
  color:#fff !important;
  -webkit-text-fill-color:#fff !important;
  opacity:1 !important;
}
[data-testid="stTextArea"] textarea {min-height:110px;}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stNumberInput"] input::placeholder,
[data-testid="stDateInput"] input::placeholder,
[data-testid="stTimeInput"] input::placeholder,
[data-testid="stChatInput"] textarea::placeholder {
  color:rgba(255,255,255,.72) !important;opacity:1;
}
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
[data-testid="stDateInput"] div[data-baseweb="input"]:focus-within,
[data-testid="stTimeInput"] div[data-baseweb="input"]:focus-within,
[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within,
[data-testid="stMultiSelect"] div[data-baseweb="select"]:focus-within,
[data-testid="stTextArea"] textarea:focus {
  border-color:var(--teal) !important;
  box-shadow:0 0 0 3px rgba(17,124,118,.13) !important;
}
[data-testid="stButton"] button,[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button,[data-testid="stFileUploaderDropzone"] button {
  min-height:44px;border:1px solid var(--line);border-radius:10px;
  background:var(--paper);color:var(--ink);font-size:15px;font-weight:600;
  font-family:'Open Sauce One',sans-serif !important;transition:background .16s ease,color .16s ease;
}
[data-testid="stButton"] button:hover,[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
  border-color:var(--teal);background:rgba(17,124,118,.08);color:var(--teal);
}
[data-testid="stButton"] button[kind="primary"],
[data-testid="stFormSubmitButton"] button[kind="primary"],
[data-testid="stDownloadButton"] button[kind="primary"] {
  background:var(--teal);border-color:var(--teal);color:var(--paper);
}
[data-testid="stButton"] button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
[data-testid="stDownloadButton"] button[kind="primary"]:hover {
  background:var(--ink);border-color:var(--ink);color:var(--paper);
}
[data-testid="stButton"] button:disabled,[data-testid="stFormSubmitButton"] button:disabled {
  background:var(--soft);border-color:var(--line);color:rgba(25,61,69,.48);
}
[data-testid="stFileUploaderDropzone"] {
  background:var(--paper) !important;border:1px dashed rgba(17,124,118,.55) !important;
  border-radius:14px !important;
}
[data-testid="stAlert"] {
  background:var(--paper) !important;color:var(--ink) !important;
  border:1px solid var(--line) !important;border-radius:12px;
}
[data-testid="stAlert"] > div:first-child {
  background:var(--paper) !important;color:var(--ink) !important;
}
[data-testid="stAlert"] p,[data-testid="stAlert"] span {
  background:transparent !important;color:var(--ink) !important;
}
[data-testid="stAlert"] svg {color:var(--teal) !important;fill:var(--teal) !important;}
[data-testid="stDataFrame"],[data-testid="stTable"] {
  background:var(--paper);border:1px solid var(--line);border-radius:12px;overflow:hidden;
}
[data-testid="stDataFrame"] *,[data-testid="stTable"] * {color:var(--ink);}
[data-testid="stTabs"] button,[data-testid="stExpander"] summary {
  color:var(--ink) !important;font-size:15px !important;
}
[data-testid="stProgress"] > div > div {background:var(--teal);}
[data-testid="stChatInput"] textarea {
  background:#161b22 !important;color:#fff !important;
  -webkit-text-fill-color:#fff !important;
  border:1px solid var(--line);border-radius:12px;font-size:15px;
}
[data-testid="stChatInput"] button {background:var(--teal) !important;color:var(--paper) !important;}
[data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"] {
  background:#161b22 !important;color:#fff !important;
}
[data-baseweb="menu"] [role="option"],[role="listbox"] [role="option"] {
  color:#fff !important;background:#161b22 !important;
}
[data-baseweb="menu"] [role="option"]:hover,
[role="listbox"] [role="option"]:hover {
  color:#fff !important;background:rgba(17,124,118,.72) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
  background:rgba(17,124,118,.72) !important;color:#fff !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] * {color:#fff !important;}
[data-testid="stChatMessage"] {
  background:var(--paper);border:1px solid var(--line);border-radius:14px;
}
.workspace-section {
  color:var(--teal);font-size:13px;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;margin:4px 0 8px;
}
.workspace-subtitle {
  color:var(--muted);font-size:16px;line-height:1.6;margin:8px 0 24px;
}
.doctor-eyebrow,.patient-eyebrow {
  color:var(--teal);font-size:13px;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;
}
.doctor-subtitle,.patient-subtitle {
  color:var(--muted);font-size:16px;line-height:1.6;margin:8px 0 24px;
}
.doctor-panel,.patient-panel {margin-bottom:18px;}
.doctor-callout,.patient-callout,.patient-warning {
  background:rgba(17,124,118,.08);color:var(--ink);border:1px solid var(--line);
  border-radius:12px;padding:16px 18px;font-size:15px;line-height:1.6;margin:14px 0;
}
.patient-metric {
  background:var(--paper);border:1px solid var(--line);border-radius:14px;
  padding:18px;min-height:96px;
}
.patient-metric-value {font-size:28px;line-height:1.2;font-weight:700;color:var(--ink);}
.patient-metric-label {font-size:14px;color:var(--muted);margin-top:7px;}
.patient-sidebar-brand,.doctor-sidebar-brand {
  display:flex;align-items:center;gap:12px;color:var(--ink);margin:10px 0 30px;
  font-size:19px;line-height:1.25;font-weight:700;
}
.patient-sidebar-brand strong,.doctor-sidebar-brand strong {white-space:nowrap;}
.patient-sidebar-brand small,.doctor-sidebar-brand small {display:none;}
.patient-brand-mark,.doctor-brand-mark {
  display:grid;place-items:center;flex:0 0 40px;width:40px;height:40px;
  border-radius:12px;background:rgba(17,124,118,.1);color:var(--teal);font-size:22px;
}
.patient-brand-mark svg,.doctor-brand-mark svg {
  display:block;width:24px;height:24px;stroke:currentColor;fill:none;
  stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;
}
.patient-sidebar-logo,.doctor-sidebar-logo {
  display:grid;place-items:center;flex:0 0 40px;width:40px;height:40px;
  border-radius:12px;background:rgba(17,124,118,.1);color:var(--teal);
}
.patient-sidebar-logo svg,.doctor-sidebar-logo svg {
  display:block;width:25px;height:25px;stroke:currentColor;fill:none;
  stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;
}
.patient-sidebar-section-label,.doctor-sidebar-section-label {
  color:var(--muted);font-size:12px;font-weight:700;letter-spacing:.06em;
  margin:0 0 12px;padding:0 8px;
}
.patient-sidebar-user,.doctor-sidebar-user {
  display:flex;align-items:center;gap:12px;border-top:1px solid var(--line);
  margin-top:24px;padding:18px 4px 8px;color:var(--ink);font-size:14px;
}
.patient-sidebar-user small,.doctor-sidebar-user small {
  display:block;color:var(--muted);font-size:13px;margin-top:3px;
}
.patient-support-link {
  display:block;margin:20px 10px 0;color:var(--teal) !important;
  font-size:14px;font-weight:600;text-decoration:none;
}
.patient-support-link:hover {text-decoration:underline;}
body:has(.doctor-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"],
body:has(.patient-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"] {
  background:var(--paper) !important;border-right:1px solid var(--line);
  min-width:264px !important;width:264px !important;max-width:264px !important;
  flex:0 0 264px !important;
}
body:has(.doctor-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"] > div:first-child,
body:has(.patient-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
  width:264px !important;padding:20px 16px;
}
body:has(.doctor-workspace-marker) [data-testid="collapsedControl"] button,
body:has(.patient-workspace-marker) [data-testid="collapsedControl"] button,
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] button[kind="header"],
body:has(.patient-workspace-marker) [data-testid="stSidebar"] button[kind="header"],
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] button[kind="headerNoPadding"],
body:has(.patient-workspace-marker) [data-testid="stSidebar"] button[kind="headerNoPadding"] {
  display:grid !important;place-items:center;min-width:40px !important;
  width:40px !important;height:40px !important;padding:0 !important;
  color:var(--ink) !important;background:var(--paper) !important;
  border:1px solid var(--line) !important;border-radius:10px !important;
  opacity:1 !important;
}
body:has(.doctor-workspace-marker) [data-testid="collapsedControl"] button svg,
body:has(.patient-workspace-marker) [data-testid="collapsedControl"] button svg,
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] button[kind="header"] svg,
body:has(.patient-workspace-marker) [data-testid="stSidebar"] button[kind="header"] svg,
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] button[kind="headerNoPadding"] svg,
body:has(.patient-workspace-marker) [data-testid="stSidebar"] button[kind="headerNoPadding"] svg {
  width:20px !important;height:20px !important;color:var(--ink) !important;
  fill:var(--ink) !important;opacity:1 !important;
}
body:has(.doctor-workspace-marker) [data-testid="collapsedControl"] button:hover,
body:has(.patient-workspace-marker) [data-testid="collapsedControl"] button:hover,
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] button[kind="header"]:hover,
body:has(.patient-workspace-marker) [data-testid="stSidebar"] button[kind="header"]:hover,
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] button[kind="headerNoPadding"]:hover,
body:has(.patient-workspace-marker) [data-testid="stSidebar"] button[kind="headerNoPadding"]:hover {
  background:var(--mist) !important;border-color:var(--teal) !important;
}
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] *,
body:has(.patient-workspace-marker) [data-testid="stSidebar"] * {
  font-family:'Open Sauce One',sans-serif !important;
}
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] button,
body:has(.patient-workspace-marker) [data-testid="stSidebar"] button {
  min-height:48px !important;border:1px solid transparent;border-radius:12px;
  background:transparent;color:var(--ink);font-size:15px;font-weight:500;
  text-align:left;padding:10px 12px;white-space:normal;
}
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] button p,
body:has(.patient-workspace-marker) [data-testid="stSidebar"] button p {
  color:inherit !important;font-size:15px !important;line-height:1.35;
  margin:0 !important;white-space:normal;
}
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] button:hover,
body:has(.patient-workspace-marker) [data-testid="stSidebar"] button:hover {
  color:var(--teal);background:rgba(17,124,118,.08);
}
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] button[kind="primary"],
body:has(.patient-workspace-marker) [data-testid="stSidebar"] button[kind="primary"] {
  color:var(--paper);background:var(--teal);border-color:var(--teal);font-weight:700;
}
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] button[kind="primary"] *,
body:has(.patient-workspace-marker) [data-testid="stSidebar"] button[kind="primary"] * {
  color:var(--paper) !important;
}
.workspace-nav-icon {
  display:grid;place-items:center;min-height:44px;color:var(--muted);
}
.workspace-nav-icon.is-active {color:var(--teal);}
.workspace-nav-icon svg {
  width:20px;height:20px;stroke:currentColor;fill:none;stroke-width:1.8;
  stroke-linecap:round;stroke-linejoin:round;
}
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.workspace-nav-icon),
body:has(.patient-workspace-marker) [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.workspace-nav-icon) {
  display:flex !important;flex-direction:row !important;align-items:center !important;
  gap:4px !important;
}
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.workspace-nav-icon) > [data-testid="column"]:first-child,
body:has(.patient-workspace-marker) [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.workspace-nav-icon) > [data-testid="column"]:first-child {
  flex:0 0 36px !important;width:36px !important;min-width:36px !important;
}
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.workspace-nav-icon) > [data-testid="column"]:last-child,
body:has(.patient-workspace-marker) [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.workspace-nav-icon) > [data-testid="column"]:last-child {
  flex:1 1 0 !important;width:auto !important;min-width:0 !important;
}
body:has(.doctor-workspace-marker) [data-testid="stButton"] button[kind="primary"] *,
body:has(.doctor-workspace-marker) [data-testid="stFormSubmitButton"] button[kind="primary"] *,
body:has(.patient-workspace-marker) [data-testid="stButton"] button[kind="primary"] *,
body:has(.patient-workspace-marker) [data-testid="stFormSubmitButton"] button[kind="primary"] *,
[data-testid="stDownloadButton"] button[kind="primary"] * {
  color:var(--paper) !important;
}
body:has(.doctor-workspace-marker) [data-testid="stSidebar"] button[kind="primary"] *,
body:has(.patient-workspace-marker) [data-testid="stSidebar"] button[kind="primary"] * {
  color:var(--paper) !important;
}
body:has(.doctor-workspace-marker) section.main > div.block-container,
body:has(.patient-workspace-marker) section.main > div.block-container {
  max-width:1560px;padding:30px 40px 56px;
  container-type:inline-size;container-name:workspace-layout;
}
body:has(.doctor-workspace-marker) [data-testid="stBottom"],
body:has(.patient-workspace-marker) [data-testid="stBottom"] {background:var(--mist) !important;}
.landing-page,.auth-page {
  color:var(--ink);font-family:'Open Sauce One',sans-serif !important;
}
.landing-page *,.auth-page * {font-family:'Open Sauce One',sans-serif !important;}
.landing-page p,.auth-page p {color:var(--muted);}
.landing-page h1,.landing-page h2,.landing-page h3,
.auth-page h1,.auth-page h2,.auth-page h3 {color:var(--ink) !important;}
.auth-page [data-testid="stFormSubmitButton"] button:disabled {
  background:var(--soft) !important;color:var(--muted) !important;
  border:1px solid var(--line) !important;
}
body:has(.auth-page) section.main > div.block-container {
  container-type:inline-size;container-name:auth-layout;
}
@container workspace-layout (max-width: 760px) {
  [data-testid="stHorizontalBlock"] {
    display:flex !important;flex-direction:column !important;align-items:stretch !important;
  }
  [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    min-width:100% !important;width:100% !important;max-width:100% !important;
    flex:0 0 auto !important;
  }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
    display:grid !important;grid-template-columns:repeat(2,minmax(0,1fr));
    gap:12px !important;
  }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="column"] {
    min-width:0 !important;width:100% !important;max-width:none !important;
  }
}
@container auth-layout (max-width: 720px) {
  body:has(.auth-page) [data-testid="stHorizontalBlock"] {
    flex-direction:column !important;gap:12px !important;
  }
  body:has(.auth-page) [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    width:100% !important;min-width:100% !important;max-width:100% !important;
    flex:0 0 auto !important;
  }
}
@media(max-width:1100px) {
  section.main > div.block-container {padding:26px 26px 42px;}
}
body:has(.admin-workspace-marker) section.main > div.block-container {
    max-width:1560px;padding:30px 40px 56px;container-type:inline-size;
}
body:has(.admin-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"] {
    background:var(--paper) !important;border-right:1px solid var(--line);
    min-width:264px !important;width:264px !important;max-width:264px !important;
}
body:has(.admin-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width:264px !important;padding:20px 16px;
}
body:has(.admin-workspace-marker) [data-testid="stSidebar"] * {
    font-family:'Open Sauce One',sans-serif !important;
}
body:has(.admin-workspace-marker) [data-testid="stSidebar"] button {
    min-height:44px !important;border:1px solid transparent;border-radius:12px;
    background:transparent;color:var(--ink);font-size:14px;font-weight:500;
    text-align:left;padding:9px 12px;
}
body:has(.admin-workspace-marker) [data-testid="stSidebar"] button:hover {
    color:var(--teal);background:rgba(17,124,118,.08);
}
body:has(.admin-workspace-marker) [data-testid="stSidebar"] button[kind="primary"] {
    color:var(--teal);background:rgba(17,124,118,.12);border-color:transparent;font-weight:700;
}
.admin-shell-title {font-size:34px;line-height:1.15;font-weight:700;color:var(--ink);margin:0;}
.admin-shell-subtitle {font-size:15px;line-height:1.5;color:var(--muted);margin:7px 0 24px;}
.admin-eyebrow {color:var(--teal);font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;}
.admin-panel {background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 8px 24px rgba(25,61,69,.05);margin-bottom:16px;}
.admin-panel h3 {margin-top:0;}
.admin-muted {color:var(--muted);font-size:13px;}
.admin-status {display:inline-block;padding:5px 9px;border-radius:999px;background:rgba(17,124,118,.1);color:var(--teal);font-size:12px;font-weight:700;}
.admin-note {background:rgba(17,124,118,.08);border:1px solid rgba(17,124,118,.12);border-radius:12px;padding:13px 15px;color:var(--ink);font-size:13px;}
.pending-workspace-marker {display:none;}
body:has(.pending-workspace-marker) section.main > div.block-container {
  max-width:1560px;padding:30px 40px 56px;container-type:inline-size;
}
body:has(.pending-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"] {
  background:var(--paper) !important;border-right:1px solid var(--line);
  min-width:178px !important;width:178px !important;max-width:178px !important;
}
body:has(.pending-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
  width:178px !important;padding:20px 15px;
}
body:has(.pending-workspace-marker) [data-testid="stSidebar"] button {
  min-height:42px !important;color:var(--ink);background:transparent;
  border:1px solid transparent;border-radius:9px;font-size:13px;padding:8px 12px;
}
body:has(.pending-workspace-marker) [data-testid="stSidebar"] button:hover {
  color:var(--teal);background:rgba(17,124,118,.08);
}
body:has(.pending-workspace-marker) [data-testid="stSidebar"] button[kind="primary"] {
  color:var(--teal);background:rgba(17,124,118,.12);border-color:transparent;font-weight:700;
}
.pending-breadcrumb {color:#8aa0a5;font-size:12px;margin:0 0 24px;}
.pending-page-title {font-size:29px !important;margin:0 0 8px;color:var(--ink) !important;}
.pending-page-subtitle {font-size:14px;color:#80979d;margin-bottom:26px;}
.pending-status-badge {position:absolute;right:42px;top:18px;border-radius:999px;background:#fff3d9;color:#a86b17;padding:8px 22px;font-size:12px;font-weight:700;}
.pending-review-card,.pending-panel,.pending-feature-card {
  background:var(--paper);border:1px solid #dfe9e8;border-radius:12px;
  padding:24px;box-shadow:none;margin-bottom:18px;color:var(--ink);
}
.pending-review-card h2 {font-size:21px !important;margin:20px 0 10px;}
.pending-review-card p,.pending-feature-card p,.pending-panel p {color:#80979d;font-size:14px;line-height:1.55;}
.pending-pill {display:inline-block;background:#fff3d9;color:#a86b17;border-radius:999px;padding:6px 20px;font-size:12px;font-weight:700;}
.pending-divider {height:1px;background:#e1eceb;margin:22px 0;}
.pending-review-grid {display:grid;grid-template-columns:1fr 1fr;gap:20px;}
.pending-review-grid small,.pending-panel > small {display:block;color:#80979d;font-size:10px;font-weight:700;margin-bottom:8px;}
.pending-review-grid strong,.pending-panel > strong {display:block;color:#315d62;font-size:13px;}
.pending-section-title {font-size:17px !important;margin:30px 0 14px;}
.pending-feature-card {min-height:112px;padding:22px 18px;}
.pending-feature-card h3,.pending-panel h3 {font-size:16px !important;margin:0 0 10px;}
.pending-feature-highlight {background:#e5f5f3;}
.pending-footnote,.pending-muted {color:#80979d !important;font-size:12px !important;}
.pending-panel h4 {color:#315d62;font-size:14px;margin:18px 0 8px;}
.pending-review-text {color:#a86b17 !important;margin-top:22px;}
.pending-banner {background:#fff3d9;color:#a86b17;border-radius:10px;padding:12px 14px;font-size:13px;margin:0 0 18px;}
.pending-chat-panel {min-height:410px;}
.pending-chat-user {background:#e5f5f3;color:#315d62;border-radius:9px;padding:15px;margin:18px 0 12px;margin-left:28%;}
.pending-chat-assistant {background:#f4f8f8;color:#315d62;border-radius:9px;padding:18px;margin:12px 0;}
.pending-privacy {background:#e5f5f3;}
body:has(.pending-workspace-marker) [data-testid="stSidebar"] .pending-sidebar-callout {
  background:#f4f8f8;border-radius:10px;padding:14px 10px;margin:220px 0 24px;
}
.pending-sidebar-callout strong {display:block;color:#315d62;font-size:11px;}
.pending-sidebar-callout small {display:block;color:#8aa0a5;font-size:10px;line-height:1.45;margin-top:8px;}
body:has(.pending-workspace-marker) [data-testid="stTextInput"] input:disabled {
  color:#fff !important;-webkit-text-fill-color:#fff !important;opacity:1 !important;
}
@media(max-width:720px) {
  section.main > div.block-container {padding:22px 18px 36px;}
  body:has(.doctor-workspace-marker) [data-testid="stVerticalBlockBorderWrapper"],
  body:has(.patient-workspace-marker) [data-testid="stVerticalBlockBorderWrapper"] {
    padding:10px !important;
  }
  body:has(.doctor-workspace-marker) [data-testid="stForm"],
  body:has(.patient-workspace-marker) [data-testid="stForm"] {
    padding:8px !important;
  }
  body:has(.doctor-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"],
  body:has(.patient-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"] {
    min-width:100vw !important;width:100vw !important;
    max-width:100vw !important;flex:0 0 100vw !important;
  }

  body:has(.doctor-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"] > div:first-child,
  body:has(.patient-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width:100vw !important;
  }
  h1 {font-size:30px !important;}
  h2 {font-size:23px !important;}
  h3 {font-size:19px !important;}
  .pending-status-badge {position:static;display:inline-block;margin:-4px 0 20px;}
  .pending-review-grid {grid-template-columns:1fr;}
  body:has(.pending-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"],
  body:has(.pending-workspace-marker) [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width:100vw !important;min-width:100vw !important;max-width:100vw !important;
  }
}
@media(max-width:420px) {
  section.main > div.block-container {padding:18px 14px 30px;}
  .app-card,.doctor-panel,.patient-panel {padding:16px;border-radius:14px;}
}
</style>
"""


def inject_theme() -> None:
    """Apply the shared light palette and Open Sauce typography."""
    st.markdown(_THEME, unsafe_allow_html=True)


def inject_doctor_workspace_theme() -> None:
    """Apply the shared clinical workspace presentation to doctor routes."""
    st.markdown(_THEME, unsafe_allow_html=True)


def inject_patient_workspace_theme() -> None:
    """Apply the shared clinical workspace presentation to patient routes."""
    st.markdown(_THEME, unsafe_allow_html=True)


def inject_admin_workspace_theme() -> None:
    """Apply the shared light workspace presentation to administrator routes."""
    st.markdown(_THEME, unsafe_allow_html=True)
