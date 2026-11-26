import json
from unittest.mock import MagicMock, patch

import pytest

from utils.web_search import search_web


def test_search_web_returns_only_https_results_from_trusted_medical_domains():
    response = MagicMock()
    response.read.return_value = json.dumps({
        "results": [
            {
                "title": "Medical source",
                "url": "https://www.cancer.gov/about-cancer/brain-tumors",
                "content": "Trusted search result.",
            },
            {
                "title": "Untrusted source",
                "url": "https://example.com/brain-tumors",
                "content": "Must not be returned.",
            },
            {
                "title": "Insecure source",
                "url": "http://www.cancer.gov/about-cancer/brain-tumors",
                "content": "Must not be returned.",
            },
        ]
    }).encode("utf-8")
    response_context = MagicMock()
    response_context.__enter__.return_value = response

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
        patch("utils.web_search.urllib.request.urlopen", return_value=response_context) as urlopen,
    ):
        results = search_web("brain tumor", max_results=10)

    request_payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
    assert request_payload["max_results"] == 5
    assert request_payload["include_domains"]
    assert results == [{
        "title": "Medical source",
        "url": "https://www.cancer.gov/about-cancer/brain-tumors",
        "content": "Trusted search result.",
    }]


def test_search_web_reports_missing_api_key():
    with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
        with pytest.raises(RuntimeError, match="not configured"):
            search_web("brain tumor")
