"""
Shared fixtures for the whole suite.

API keys are read from the environment when settings.py is imported and
validated by the services before a run. The suite must not depend on a
developer's .env being discoverable, so every test sees fake keys unless it
patches its own. Tests that hit no LLM never notice; the LLM itself is
mocked or stubbed wherever a request would be sent.
"""

import pytest


@pytest.fixture(autouse=True)
def _fresh_prompt_cache_state():
    """The marker verdict and the live cache counters are process-wide; a test
    that taught them must not change what the next test sends or logs."""
    from claimlens.llmclient import LLMClient
    LLMClient._CACHE_MARKER_CACHE.clear()
    LLMClient._CACHE_WATCH.clear()
    yield
    LLMClient._CACHE_MARKER_CACHE.clear()
    LLMClient._CACHE_WATCH.clear()


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch):
    monkeypatch.setattr("claimlens.settings.EXTRACTOR_API_KEY", "test-key")
    monkeypatch.setattr("claimlens.settings.CHECKER_API_KEY", "test-key")
    monkeypatch.setattr("claimlens.settings.ATOMIZER_API_KEY", "test-key")
