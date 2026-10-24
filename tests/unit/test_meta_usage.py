"""_meta.usage — what one report cost, not what the process has cost so far.

GLOBAL_STATS never resets, so a raw read under --runs would bill run 3 for
runs 1 and 2. Every envelope report snapshots at its start and writes the
delta; the --runs wrappers sum the run docs.
"""

from unittest.mock import patch

import pytest

from claimlens.stats import GLOBAL_STATS, sum_usage, usage_since


def _bump(phase: str, requests: int, tokens_in: int, tokens_out: int = 1, answered: bool = True):
    GLOBAL_STATS.set_phase(phase)
    for _ in range(requests):
        GLOBAL_STATS.log_request()
        if answered:
            GLOBAL_STATS.update({"prompt_tokens": tokens_in, "completion_tokens": tokens_out})


class TestDelta:

    def test_only_what_happened_after_the_snapshot(self):
        _bump("extract", 5, 100)
        before = GLOBAL_STATS.snapshot()
        _bump("extract", 3, 100)
        u = usage_since(before)
        assert u["requests"] == 3
        assert u["input_tokens"] == 300
        assert u["phases"]["extract"]["requests"] == 3

    def test_nothing_happened_gives_zero_and_no_phases(self):
        before = GLOBAL_STATS.snapshot()
        assert usage_since(before) == {"requests": 0, "input_tokens": 0,
                                       "output_tokens": 0, "reasoning_tokens": 0,
                                       "cached_tokens": None}

    def test_phase_untouched_since_snapshot_is_omitted(self):
        _bump("atomize", 4, 50)
        before = GLOBAL_STATS.snapshot()
        _bump("check", 2, 70)
        assert set(usage_since(before)["phases"]) == {"check"}

    def test_timeouts_are_requests_without_tokens(self):
        """The doc promise: a call that never answered still counts as a request."""
        before = GLOBAL_STATS.snapshot()
        _bump("check", 2, 0, answered=False)
        u = usage_since(before)
        assert u["requests"] == 2
        assert u["input_tokens"] == 0


class TestSum:

    def test_adds_totals_and_phases(self):
        a = {"requests": 2, "input_tokens": 10, "output_tokens": 1, "reasoning_tokens": 0,
             "phases": {"extract": {"requests": 2, "input_tokens": 10, "output_tokens": 1, "reasoning_tokens": 0}}}
        b = {"requests": 3, "input_tokens": 20, "output_tokens": 2, "reasoning_tokens": 5,
             "phases": {"check": {"requests": 3, "input_tokens": 20, "output_tokens": 2, "reasoning_tokens": 5}}}
        s = sum_usage([a, b])
        assert s["requests"] == 5 and s["input_tokens"] == 30 and s["reasoning_tokens"] == 5
        assert set(s["phases"]) == {"extract", "check"}

    def test_tolerates_runs_without_usage(self):
        """Older run docs (and the fake reports in test_runs) carry no usage."""
        assert sum_usage([None, {}, {"requests": 1, "input_tokens": 2,
                                     "output_tokens": 3, "reasoning_tokens": 0}])["requests"] == 1


class TestCachedTokens:
    """Prompt-cache reads: the standard field only, null when unknown, never 0."""

    def _answer(self, phase: str, usage: dict):
        GLOBAL_STATS.set_phase(phase)
        GLOBAL_STATS.log_request()
        GLOBAL_STATS.update(usage)

    def test_reported_reads_are_summed_per_phase(self):
        before = GLOBAL_STATS.snapshot()
        for cached in (0, 900):
            self._answer("extract", {"prompt_tokens": 1000, "completion_tokens": 1,
                                     "prompt_tokens_details": {"cached_tokens": cached}})
        u = usage_since(before)
        assert u["cached_tokens"] == 900
        assert u["phases"]["extract"]["cached_tokens"] == 900

    def test_a_reported_miss_is_zero(self):
        before = GLOBAL_STATS.snapshot()
        self._answer("check", {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}})
        assert usage_since(before)["cached_tokens"] == 0

    @pytest.mark.parametrize("usage", [
        {"prompt_tokens": 10},                                              # no details (Ollama)
        {"prompt_tokens": 10, "prompt_tokens_details": None},               # null details
        {"prompt_tokens": 10, "prompt_tokens_details": {}},                 # empty (Anthropic compat)
        {"prompt_tokens": 10, "prompt_cache_hit_tokens": 8},                # DeepSeek-direct naming
        {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": "8"}},
        {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": True}},
        {"prompt_tokens": 10, "prompt_tokens_details": "garbage"},
    ])
    def test_anything_else_is_unknown_not_zero(self, usage):
        before = GLOBAL_STATS.snapshot()
        self._answer("check", usage)
        u = usage_since(before)
        assert u["cached_tokens"] is None
        assert u["input_tokens"] == 10            # the rest of the tally is untouched

    def test_sum_keeps_null_unless_a_run_reported(self):
        base = {"requests": 1, "input_tokens": 10, "output_tokens": 1, "reasoning_tokens": 0}
        assert sum_usage([{**base, "cached_tokens": None}] * 2)["cached_tokens"] is None
        assert sum_usage([{**base, "cached_tokens": None}, {**base, "cached_tokens": 7}])["cached_tokens"] == 7


class TestTokenTable:
    """The 📊 Tokens line per command carries the cache read, or says it can't."""

    def _line(self, caplog, phase: str) -> str:
        from claimlens.stats import log_token_stats
        caplog.clear()
        with caplog.at_level("INFO"):
            log_token_stats()
        return next(r.getMessage() for r in caplog.records if f" {phase}:" in r.getMessage())

    def test_cache_read_with_percent(self, caplog):
        GLOBAL_STATS.set_phase("tt_hit")
        GLOBAL_STATS.log_request()
        GLOBAL_STATS.update({"prompt_tokens": 1000, "completion_tokens": 5,
                             "prompt_tokens_details": {"cached_tokens": 900}})
        assert "1,000 in (900 cache read · 90%) · 5 out" in self._line(caplog, "tt_hit")

    def test_not_reported(self, caplog):
        GLOBAL_STATS.set_phase("tt_unknown")
        GLOBAL_STATS.log_request()
        GLOBAL_STATS.update({"prompt_tokens": 1000, "completion_tokens": 5})
        assert "1,000 in (cache use not reported by endpoint) · 5 out" in self._line(caplog, "tt_unknown")

    def test_no_answered_tokens_says_nothing_about_cache(self, caplog):
        GLOBAL_STATS.set_phase("tt_silent")
        GLOBAL_STATS.log_request()                 # a timeout: request, no usage
        assert "0 in · 0 out" in self._line(caplog, "tt_silent")


class TestWiring:

    def test_refcheck_meta_bills_only_its_own_run(self):
        from claimlens.pipelines.refchecker import RefCheckerPipeline

        _bump("extract", 7, 100)            # earlier work in the same process

        with patch("claimlens.pipelines.refchecker.ExtractionService"), \
             patch("claimlens.pipelines.refchecker.CheckingService"):
            p = RefCheckerPipeline(extractor_model="ext-model",
                                   checker_model="chk-model", verbosity="silent")

        async def ext_run(data):
            _bump("extract", 2, 50)
            return data

        async def chk_run(data):
            _bump("check", 1, 30)
            return data
        p._extraction.run = ext_run
        p._checking.run = chk_run

        p.run_sync([{"response": "r", "reference": ["ref"]}])
        u = p.last_report["_meta"]["usage"]
        assert u["requests"] == 3
        assert u["phases"]["extract"]["requests"] == 2
        assert u["phases"]["check"]["requests"] == 1

    def test_runs_wrapper_sums_the_run_docs(self, monkeypatch):
        from claimlens.pipelines.ragchecker import RagCheckerPipeline

        with patch("claimlens.services.extraction.Extractor"), \
             patch("claimlens.services.checking.Checker"):
            p = RagCheckerPipeline(extractor_model="ext-model", checker_model="chk-model",
                                   runs=3, verbosity="silent")

        async def fake_run_once(data, announce=True, report=True):
            before = GLOBAL_STATS.snapshot()
            _bump("extract", 2, 50)
            p.last_run = {"_meta": {"usage": usage_since(before)},
                          "metrics": {}, "counts": {}, "items": []}
            p.last_run_findings = {"_meta": {}, "findings": {}}
            return data

        monkeypatch.setattr(p, "_run_once", fake_run_once)
        p.run_sync([{"response": "x"}])
        doc = p.last_report
        assert [r["_meta"]["usage"]["requests"] for r in doc["runs"]] == [2, 2, 2]
        assert doc["_meta"]["usage"]["requests"] == 6
        assert doc["_meta"]["usage"]["phases"]["extract"]["input_tokens"] == 300
