# 🧠 NeuroScan AI — Advanced Brain Tumor Detection & Clinical Diagnostics Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow 2.15](https://img.shields.io/badge/ML-TensorFlow%202.15-FF6F00.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI Tests](https://github.com/nmit-1NT23CS269/Brain-Tumour-Detection/actions/workflows/tests.yml/badge.svg)](https://github.com/nmit-1NT23CS269/Brain-Tumour-Detection/actions/workflows/tests.yml)

**NeuroScan AI** is a medical-grade Brain MRI Analysis and Clinical Workflow platform designed for high-accuracy brain tumor detection, deep learning explainability (Grad-CAM), OpenCV tumor segmentation overlays, specialist doctor-patient collaboration, and AI-assisted diagnostics.

---

## 🌟 Key System Capabilities

### 1. 🔬 Deep Learning MRI Classification & Explainability
- **4-Class Tumour Detection**: Classifies MRI scans into **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor (Normal)**.
- **Multi-Model Transfer Learning Registry**: Supports `VGG16`, `ResNet50`, `EfficientNetB0`, and `MobileNetV2`.
- **Grad-CAM Visual Heatmaps**: Highlights exact neural focus areas driving the model's classification decision.
- **OpenCV Tumor Segmentation**: Auto-extracts tumor boundaries, contour masks, and calculates percentage area coverage.
- **Automated PDF Diagnostic Reports**: Generates comprehensive PDF reports including original MRI, preprocessed scan, Grad-CAM heatmap, tumor segmentation overlay, AI summary, and doctor clinical recommendations.

### 2. 🤖 Mastishka AI Medical Assistant
- **ChatGPT-Style Conversational AI (`chatbot/`)**: Provider-agnostic LLM integration (OpenAI, Gemini, Anthropic, local Ollama) combined with Context-Aware chat history for open-ended, natural language Q&A.
- **Hybrid RAG Retriever (`retriever.py`)**: Combines 50% keyword overlap (with synonym expansion for terms like *tumour*, *mri*, *head pain*, *cure*) and 50% TF-IDF cosine similarity (word 1–2 n-grams + char_wb 3–4 n-grams) over NCI/NINDS knowledge base CSVs.
- **Patient Privacy & PII Sanitization (`llm_client.py`)**: Automatically strips patient codes (`PAT-XXXXXX`), email addresses, and phone numbers before sending prompts to external LLM providers.
- **Medical Safety Protocols**: Immediate emergency keyword interception (redirecting crisis queries to urgent medical care), self-diagnosis safeguards ("do I have a tumor?"), and mandatory medical disclaimers for Treatment, Prognosis, Symptoms, Medical Guidance, and Risk Factors.
- **Optional online AI and cited research**: Mastishka can send opted-in questions to the configured cloud LLM and search trusted medical websites through Tavily. PDF reports can add an AI-written educational context section with source links; model output is not treated as a clinical finding.
- **Offline Fallback & Citation**: Runs seamlessly offline using local knowledge base retrieval when no LLM API key is present. Automatically cites authoritative sources (*Sources: NCI, NINDS*) and supports dynamic scan result explanations.

### 3. 👨‍⚕️ Mandatory Doctor Credential Upload & Admin Verification
- **Doctor Registration Security**: Doctors must upload valid medical credentials (**Medical Degree Certificate** and **Medical Registration License**) in PDF, PNG, or JPG formats.
- **Multi-State Application Lifecycle**: Applications progress through `PENDING` → Admin Review → `APPROVED` / `ACTIVE`, `REJECTED`, or `ADDITIONAL_INFO_REQUIRED`.
- **Admin Verification Console**: Administrators can inspect uploaded PDF degree/license certificates, download credentials, approve applications, or request document replacements with custom feedback notes.

### 4. 📋 Doctor Add Patient & Scan Patient Workflows
- **Unique Patient Identifier (`PAT-XXXXXX`)**: Every patient receives an immutable backend-generated patient code (e.g. `PAT-000014`).
- **`+ Add Patient`**:
  - **Search Existing Patient**: Search by name, email, phone, or patient code (`PAT-XXXXXX`) and send an invitation without granting access.
  - **Register New Patient**: Create a patient account and send a relationship invitation. New patients set a password and verify their phone before accepting.
  - **Patient consent**: Doctor-created relationships remain pending until the patient explicitly accepts. Invitation tokens are stored as hashes and expire after seven days.
- **`🔬 Scan Patient`**: Approved doctors can scan JPG or PNG MRI images (up to 10 MB) only for patients with an accepted relationship. Results are linked to that patient's account.
- **Strict Backend Authorization**: Scan, report, and clinical-note access is checked against the active patient relationship; pending, declined, expired, or revoked invitations do not grant access.

### 5. 👥 Role-Based Access Control (RBAC) & Secure Auth
- **Patient Workspace**: Upload MRI scans, view Grad-CAM heatmaps, download PDF reports, assign specialist doctors, and view attached doctor clinical notes.
- **Doctor Clinical Workspace**: Manage the patient roster, review scans, export reports, and save clinician-authored observations and recommendations. Saving a note does not constitute a formal digital signature.
- **Admin Console**: Doctor verification portal, user account management, platform metrics, and immutable audit logging (`audit_logs`).
- **Simulated OTP Flow**: Phone authentication uses simulated OTP delivery (logged to server logs for dev/testing; ready for SMS provider integration in production).

---

## 🔑 Demo Accounts for Faculty & Reviewers

To make evaluation simple and seamless, **NeuroScan AI** includes a demonstration environment controlled by `DEMO_MODE=true` (defaults to `false` for production security).

To enable quick-login buttons for evaluation, copy `.env.example` to `.env` and set `DEMO_MODE=true`.

Below are the pre-configured **Demo Credentials**:

| User Role | Email Address | Default Password | Account Status | Pre-Populated Demo Content |
|---|---|---|---|---|
| **👤 Demo Patient** | `patient.demo@example.com` | `Patient@123` | `ACTIVE` | Demo patient account. Doctor access requires an accepted invitation. |
| **👨‍⚕️ Verified Doctor** | `doctor.demo@example.com` | `Doctor@123` | `APPROVED` | Demo credential documents; patient access requires a patient-accepted invitation. |
| **⏳ Pending Doctor** | `pending.doctor@example.com` | `Doctor@123` | `PENDING` | `MBBS, MD Pediatric Neurology`, `NMC-PENDING-7712`, uploaded PDF verification certificates for testing the Admin Verification & Approval workflow. |
| **⚙️ Demo Admin** | `admin.demo@example.com` | `Admin@123` | `ACTIVE` | Full administrative privileges, Doctor Approval Queue, User Control, Metrics, & Audit Trail. |

> ⚡ **One-Click Quick Login**: When `DEMO_MODE=true`, the Sign In page renders one-click login buttons to authenticate instantly into standard SQLite RBAC accounts!

---

## 📁 Repository Structure

```text
brain_tumour_detection/
├── .github/
│   └── workflows/
│       └── tests.yml           # GitHub Actions CI automated test workflow
├── app.py                      # Main Streamlit web application & role routing
├── authentication/             # Password hashing, bcrypt verification, OTP, & RBAC auth
├── backend/                    # Core scan service execution & ML inference handler
├── chatbot/                    # Mastishka AI Medical Assistant module & knowledge base
├── data/                       # Raw & processed MRI dataset directory (see DATASET.md)
├── database/                   # SQLite database schema, CRUD helpers, & migration engine
├── explainability/             # Grad-CAM heatmap generation module
├── frontend/                   # Custom dark-theme CSS styling & UI layout tiles
├── model/                      # Transfer learning CNN model architecture wrappers
├── preprocessing/              # Image normalization, resizing, quality checks, & histograms
├── reports/                    # FPDF automated PDF diagnostic report generator
├── segmentation/               # OpenCV contour extraction & tumor area calculation
├── test_images/                # Sample MRI images for smoke testing
├── tests/                      # Pytest automated test suite (test_smoke.py)
├── uploads/                    # Local storage for uploaded doctor medical certificates (git-ignored)
├── .env.example                # Template for environment configuration
├── DATASET.md                  # Dataset download & extraction layout instructions
├── DEPLOYMENT.md               # Production deployment instructions
├── LICENSE                     # MIT License file
├── README.md                   # Complete project documentation
├── requirements.txt            # Python dependencies (including pytest)
├── SETUP.md                    # Environment setup guide
└── TRAINING.md                 # Model training workflow guide
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python **3.10** or **3.11** (Recommended for TensorFlow 2.15 compatibility)
- Git

### 2. Installation

Clone the repository and set up a virtual environment:

```powershell
# Clone the repository
git clone https://github.com/nmit-1NT23CS269/Brain-Tumour-Detection.git
cd Brain-Tumour-Detection

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Upgrade pip & install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download Dataset (Optional for training)

Follow instructions in [DATASET.md](DATASET.md) to download the Kaggle Brain Tumor MRI dataset into `data/raw/archive/`.

### 4. Launch Application

Run the Streamlit server:

```powershell
python -m streamlit run app.py
```

Open your browser and navigate to: **`http://localhost:8501`**

---

## ⚙️ Environment Configuration

Copy `.env.example` to `.env` to configure environment variables:

| Variable | Default | Description |
|---|---|---|
| `DEMO_MODE` | `false` | Set to `true` to render One-Click Demo Login buttons on the Sign In page. Keep `false` for production builds. |
| `DEMO_PATIENT_EMAIL` | `patient.demo@example.com` | Email address for demo patient account. |
| `DEMO_PATIENT_PASSWORD` | `Patient@123` | Password for demo patient account. |
| `DEMO_DOCTOR_EMAIL` | `doctor.demo@example.com` | Email address for demo doctor account. |
| `DEMO_DOCTOR_PASSWORD` | `Doctor@123` | Password for demo doctor account. |
| `DEMO_ADMIN_EMAIL` | `admin.demo@example.com` | Email address for demo admin account. |
| `DEMO_ADMIN_PASSWORD` | `Admin@123` | Password for demo admin account. |
| `APP_ENV` | `development` | Use `production` to disable development-mode OTP and invitation logging. |
| `APP_BASE_URL` | `http://localhost:8501` | Public app URL used to construct patient invitation links. |
| `SMTP_SERVER` | `smtp.gmail.com` | SMTP host for patient invitation email. |
| `SMTP_PORT` | `587` | SMTP port for patient invitation email. |
| `SMTP_USERNAME` | `""` | SMTP account used to send patient invitations. |
| `SMTP_PASSWORD` | `""` | SMTP credential; keep it in `.env` or a secret manager, never in source control. |
| `SMTP_FROM_EMAIL` | `SMTP_USERNAME` | Optional sender address for invitation email. |
| `LLM_PROVIDER` | `""` (empty) | LLM provider (`openai`, `gemini`, `anthropic`, `ollama`, or empty for local offline fallback mode). |
| `LLM_API_KEY` | `""` | API key for the selected cloud LLM provider. |
| `LLM_MODEL` | `""` | Model identifier (e.g., `gemini-1.5-flash`, `gpt-3.5-turbo`, `claude-3-haiku-20240307`, `llama3`). |
| `LLM_TIMEOUT` | `20` | Maximum request timeout in seconds for LLM calls. |
| `RETRIEVAL_THRESHOLD` | `0.25` | Minimum hybrid confidence score threshold required for knowledge retrieval. |
| `TAVILY_API_KEY` | `""` | Tavily Search API key for trusted medical web results and report citations. |

Set `LLM_PROVIDER` and `LLM_API_KEY` for cloud-generated answers and reports, and set `TAVILY_API_KEY` to enable live web search and citations. Chat users must opt in to online processing in Mastishka. Report research sends only the model class, confidence, class probabilities, and existing summary; MRI images and patient identifiers are not sent to search. Cloud providers process data according to their own terms, so do not enter identifying or sensitive information. Without these settings, reports disclose that they use the local model summary and Mastishka retains its offline knowledge-base mode.

---

## 🧪 Automated Testing & CI

Execute the pytest suite to verify preprocessing, model prediction, authentication security rules, and Mastishka chatbot retrieval:

```powershell
# Run full test suite (smoke tests + chatbot QA tests)
pytest -v tests/test_smoke.py tests/test_chatbot.py

# Rebuild chatbot knowledge base
python chatbot/build_knowledge_base.py
```

Automated testing is integrated with **GitHub Actions CI** (`.github/workflows/tests.yml`), running tests automatically on every push and pull request.

---

## ⚠️ Medical Disclaimer

This application is designed for **educational, capstone demonstration, and research purposes only**. It should not be used as a primary diagnostic tool or as a replacement for professional clinical evaluation by certified medical specialists.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more details.
