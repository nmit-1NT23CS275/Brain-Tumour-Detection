# Deployment Guide

## Local deployment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Push the project to a Git repository.
2. Ensure `requirements.txt` is committed.
3. Ensure `runtime.txt` contains the supported Python major/minor version (`3.11`).
4. Set the entrypoint to `app.py`.
5. Upload or mount the trained model and required artifacts.

When `models/exports/best_model.keras` is missing, MRI scan controls are disabled.
Do not enable scan predictions using an untrained fallback model.

The `runtime.txt` pin is required because the application uses TensorFlow 2.15.0,
which does not provide a Python 3.14 wheel. `setuptools<81` is retained as a
compatibility guard for build environments that still import `pkg_resources`;
PyArrow 15.0.2 is installed from its Python 3.11 wheel rather than built from
source.

## Vercel compatibility

This repository is a Streamlit application, not a Flask application. The top-level
`app.py` intentionally defines Streamlit pages and `main()`; it does not define a
Flask WSGI object named `app`, `application`, or `handler`. Therefore Vercel's
Flask/Python entrypoint detection error cannot be fixed by adding an alias without
replacing the existing application architecture and UI.

Vercel is not a supported deployment target for the current application because:

- Streamlit requires a long-lived application server and WebSocket session model.
- The project pins TensorFlow 2.15, while current Vercel Python runtimes target newer
  Python versions and serverless function bundle limits apply.
- No trained `models/exports/best_model.keras` artifact is tracked in this repository.
  The application cannot provide production inference until that artifact is supplied
  through a suitable deployment volume or inference service.

Deploy this project with Streamlit Community Cloud or a persistent Python host. A
Vercel deployment requires a separate Flask/API migration and a separately hosted
inference service; that migration must not be presented as a configuration-only fix.

## Server deployment

Use a Windows or Linux host with:

- Python `3.10` or `3.11`
- TensorFlow `2.15.x`
- Enough RAM for transfer-learning inference and training

Run with:

```powershell
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Production notes

- Replace the simulated demo OTP flow in authentication/auth.py with an integration to a real SMS gateway provider (e.g. Twilio, MSG91, AWS SNS) before production deployment.
- Back up `brain_tumour.db`.
- Keep trained model files under `models/exports/`.
- Store sensitive configuration in environment variables.
