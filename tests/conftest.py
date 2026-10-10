import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any accidental OpenAI call hits a dead local port instead of the real API."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
