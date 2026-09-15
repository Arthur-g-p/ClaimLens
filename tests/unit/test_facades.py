"""
The public API in claimlens/__init__.py: the five verbs mirror the CLI's
constructor calls, return what the CLI writes, and refuse politely inside a
running event loop. No LLM: the pipeline classes are replaced by fakes.
"""

import asyncio
import subprocess
import sys

import pytest

import claimlens


class _FakePipeline:
    """Records how it was constructed; produces a canned report."""
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.ran = None
        self.last_report = {"_meta": {"report_type": "fake"}, "metrics": {"f1": 1.0},
                            "runs": [{"items": [{"faithfulness": 1.0}]}]}
        self.last_findings = {"_meta": {"report_type": "fake"}, "runs": []}
        _FakePipeline.instances.append(self)

    def run_sync(self, data):
        self.ran = data
        return data

    async def run(self, data):
        self.ran = data
        return data


class _FakeService(_FakePipeline):
    def run_sync(self, data):
        self.ran = data
        return [dict(item, enriched=True) for item in data]


ITEMS = [{"response": "r", "gt_answer": "g", "retrieved_context": ["c"]}]


@pytest.fixture(autouse=True)
def _reset():
    _FakePipeline.instances.clear()


class TestSurface:

    def test_public_names(self):
        assert set(claimlens.__all__) >= {
            "ragcheck", "faithcheck", "refcheck", "extract", "check",
            "check_faithfulness", "acheck_faithfulness", "enable_logging", "__version__"}

    def test_version_matches_installed_metadata(self):
        from importlib.metadata import version
        assert claimlens.__version__ == version("claimlens")

    def test_import_is_lazy(self):
        # ``import claimlens`` must not drag in the LLM stack.
        code = "import sys, claimlens; print('litellm' in sys.modules, 'openai' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "False False"


class TestBatchVerbs:

    def test_ragcheck_mirrors_cli_wiring_and_returns_both_documents(self, monkeypatch):
        monkeypatch.setattr("claimlens.pipelines.ragchecker.RagCheckerPipeline", _FakePipeline)
        record, findings = claimlens.ragcheck(ITEMS, extractor_model="e", checker_model="c", concurrency=3)
        p = _FakePipeline.instances[0]
        assert p.kwargs == {"extractor_model": "e", "checker_model": "c", "concurrency": 3}
        assert p.ran is ITEMS
        assert record["metrics"]["f1"] == 1.0
        assert record["_args"] == {"command": "ragcheck", "extractor_model": "e", "checker_model": "c",
                                   "concurrency": 3,
                                   "_explicit": ["checker_model", "concurrency", "extractor_model"]}
        assert findings["_args"] == record["_args"]
        assert isinstance((record, findings), tuple)

    @pytest.mark.parametrize("verb,target", [
        ("faithcheck", "claimlens.pipelines.faithfulness.FaithfulnessPipeline"),
        ("refcheck", "claimlens.pipelines.refchecker.RefCheckerPipeline"),
    ])
    def test_other_pipelines(self, monkeypatch, verb, target):
        monkeypatch.setattr(target, _FakePipeline)
        record, _ = getattr(claimlens, verb)(ITEMS, extractor_model="e", checker_model="c")
        assert record["_args"]["command"] == verb
        assert _FakePipeline.instances[0].kwargs == {"extractor_model": "e", "checker_model": "c"}

    def test_extract_maps_uniform_names_onto_the_service(self, monkeypatch):
        monkeypatch.setattr("claimlens.services.extraction.ExtractionService", _FakeService)
        out = claimlens.extract(ITEMS, extractor_model="e", extractor_base_url="http://x", dedup=False)
        assert _FakePipeline.instances[0].kwargs == {"model": "e", "base_url": "http://x", "dedup": False}
        assert out[0]["enriched"] is True

    def test_check_maps_uniform_names_onto_the_service(self, monkeypatch):
        monkeypatch.setattr("claimlens.services.checking.CheckingService", _FakeService)
        claimlens.check(ITEMS, checker_model="c", extractor_model="e", checker_base_url="http://y")
        assert _FakePipeline.instances[0].kwargs == {"model": "c", "extractor_model": "e",
                                                     "base_url": "http://y"}


class TestEventLoop:

    def test_sync_verb_refuses_inside_a_running_loop(self, monkeypatch):
        monkeypatch.setattr("claimlens.pipelines.ragchecker.RagCheckerPipeline", _FakePipeline)

        async def inside():
            with pytest.raises(RuntimeError, match=r"await RagCheckerPipeline\(\.\.\.\)\.run\(items\)"):
                claimlens.ragcheck(ITEMS, extractor_model="e", checker_model="c")

        asyncio.run(inside())
        assert _FakePipeline.instances == []     # refused before constructing anything

    def test_realtime_sync_form_points_at_its_async_twin(self, monkeypatch):
        monkeypatch.setattr("claimlens.pipelines.faithfulness.FaithfulnessPipeline", _FakePipeline)

        async def inside():
            with pytest.raises(RuntimeError, match="acheck_faithfulness"):
                claimlens.check_faithfulness("resp", ["chunk"], extractor_model="e", checker_model="c")

        asyncio.run(inside())

    def test_async_twin_works_inside_a_loop(self, monkeypatch):
        monkeypatch.setattr("claimlens.pipelines.faithfulness.FaithfulnessPipeline", _FakePipeline)

        async def inside():
            return await claimlens.acheck_faithfulness("resp", ["chunk"], extractor_model="e", checker_model="c")

        entry = asyncio.run(inside())
        assert entry == {"faithfulness": 1.0}
        p = _FakePipeline.instances[0]
        assert p.kwargs["verbosity"] == "silent"
        assert p.ran == [{"response": "resp", "retrieved_context": ["chunk"]}]
