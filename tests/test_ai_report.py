from unittest.mock import MagicMock, patch

from reports.ai_report import generate_online_ai_report


def test_online_report_uses_trusted_search_context_and_returns_sources():
    llm_client = MagicMock()
    llm_client.is_available.return_value = True
    llm_client.generate.return_value = "General educational context [1]."
    sources = [{
        "title": "Brain tumor information",
        "url": "https://www.cancer.gov/about-cancer/brain-tumors",
        "content": "General information from NCI.",
    }]

    with (
        patch("reports.ai_report.get_llm_client", return_value=llm_client),
        patch("reports.ai_report.is_web_search_available", return_value=True),
        patch("reports.ai_report.search_web", return_value=sources) as search,
    ):
        result = generate_online_ai_report(
            prediction="Glioma",
            confidence=0.91,
            probabilities={"Glioma": 0.91, "Normal": 0.09},
            base_summary="Model-generated screening summary.",
        )

    assert result == {"summary": "General educational context [1].", "sources": sources}
    assert "Glioma" in search.call_args.args[0]
    system_prompt, messages = llm_client.generate.call_args.args
    assert "not a radiology report or diagnosis" in system_prompt
    assert "Do not infer tumor size" in system_prompt
    assert "WEB" not in messages[0]["content"]
    assert "Model-predicted class: Glioma" in messages[0]["content"]


def test_online_report_requires_web_search_configuration():
    llm_client = MagicMock()
    llm_client.is_available.return_value = True

    with (
        patch("reports.ai_report.get_llm_client", return_value=llm_client),
        patch("reports.ai_report.is_web_search_available", return_value=False),
    ):
        try:
            generate_online_ai_report("Glioma", 0.8, {"Glioma": 0.8}, "Summary")
        except RuntimeError as exc:
            assert "web search is not configured" in str(exc)
        else:
            raise AssertionError("Expected missing search configuration to be reported")
