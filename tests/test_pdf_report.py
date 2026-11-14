import numpy as np

from reports import generate_pdf_report


def test_pdf_report_includes_clinician_notes_with_untrusted_text():
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    pdf = generate_pdf_report(
        username="Patient",
        scan_date="2026-01-01",
        original_img=image,
        preprocessed_img=image,
        gradcam_img=None,
        prediction="Meningioma",
        confidence=0.8,
        probabilities={"Glioma": 0.1, "Meningioma": 0.8, "Pituitary": 0.05, "Normal": 0.05},
        class_info={"description": "Model output for clinician review.", "severity": "Review"},
        scan_id=1,
        recommendations=["No clinician recommendation has been recorded."],
        clinical_notes=[
            {
                "doctor_name": "Dr. <Example>",
                "created_at": "2026-01-01",
                "note_content": "Review & correlate.",
                "recommendation": "Follow-up <when clinically indicated>.",
            }
        ],
    )

    assert pdf.startswith(b"%PDF")
