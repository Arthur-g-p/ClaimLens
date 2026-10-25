"""--prompts: your own prompt file, over the shipped prompt_map.json.

The file holds only the keys it changes; every other key comes from the
shipped file. The entry point (CLI or library verb) resolves the path once —
fail early: unknown key, broken JSON, non-string template — and everything
below receives the template dict, passed down like --concurrency. Its hash
lands in _meta so a record still names the template set that produced it
after the file is edited; the path itself is in _args.
"""

import importlib
import inspect
import json
import re
from unittest.mock import patch

import pytest

from claimlens import settings
from claimlens.settings import describe_prompts, prompts_sha256, resolve_prompts


def _write(tmp_path, data, name="mine.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


def _custom(tmp_path):
    return resolve_prompts(_write(tmp_path, {"extractor_prompt": "Mine: {{text}}"}))


class TestResolve:

    def test_default_is_the_shipped_file(self):
        assert resolve_prompts() is settings.PROMPTS

    def test_file_replaces_only_its_keys(self, tmp_path):
        p = _custom(tmp_path)
        assert p["extractor_prompt"] == "Mine: {{text}}"
        assert p["checker_prompt"] == settings.PROMPTS["checker_prompt"]
        assert set(p) == set(settings.PROMPTS)

    def test_an_empty_file_is_the_shipped_set(self, tmp_path):
        p = resolve_prompts(_write(tmp_path, {}))
        assert p == settings.PROMPTS
        assert prompts_sha256(p) == prompts_sha256(settings.PROMPTS)

    def test_hash_follows_the_content_not_the_path(self, tmp_path):
        a = resolve_prompts(_write(tmp_path, {"extractor_prompt": "x"}, "a.json"))
        b = resolve_prompts(_write(tmp_path, {"extractor_prompt": "x"}, "b.json"))
        c = resolve_prompts(_write(tmp_path, {"extractor_prompt": "y"}, "c.json"))
        assert prompts_sha256(a) == prompts_sha256(b) != prompts_sha256(c)
        assert prompts_sha256(a) != prompts_sha256(resolve_prompts())
        assert len(prompts_sha256(a)) == 12

    def test_config_line_says_shipped_or_custom_with_the_hash(self, tmp_path):
        shipped, custom = resolve_prompts(), _custom(tmp_path)
        assert describe_prompts(shipped) == f"{settings.PROMPT_PATH} · sha256 {prompts_sha256(shipped)}"
        assert describe_prompts(custom) == f"custom, over prompt_map.json · sha256 {prompts_sha256(custom)}"

    @pytest.mark.parametrize("content, message", [
        ({"extractor_promt": "typo"}, "unknown prompt key(s) extractor_promt"),
        ({"extractor_prompt": 3}, "must be strings: extractor_prompt"),
        (["not", "an", "object"], "expected a JSON object"),
        ("{broken", "not valid JSON"),
    ])
    def test_bad_files_fail_before_any_request(self, tmp_path, content, message):
        with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
            resolve_prompts(_write(tmp_path, content))

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="Prompt file not found"):
            resolve_prompts(tmp_path / "nope.json")


class TestWorkersUseTheTemplatesTheyWereGiven:

    def test_extractor(self, tmp_path):
        from claimlens.workers.extractor import Extractor
        w = Extractor(api_key="k", model="m", base_url="http://fake/v1", prompts=_custom(tmp_path))
        assert w._build_messages("Goethe.")[-1]["content"] == "Mine: Goethe."

    def test_checker_joint(self, tmp_path):
        from claimlens.workers.checker import Checker
        t = resolve_prompts(_write(tmp_path, {"checker_prompt_joint": "J {{claims}} | {{reference}}"}))
        w = Checker(api_key="k", model="m", base_url="http://fake/v1", prompts=t)
        assert w._build_joint_messages([(1, "c")], ["r"])[-1]["content"].startswith("J [1] c | ")

    def test_atomizer(self, tmp_path):
        from claimlens.workers.atomizer import Atomizer
        t = resolve_prompts(_write(tmp_path, {"atomizer_prompt": "A {{subject}}"}))
        w = Atomizer(api_key="k", model="m", base_url="http://fake/v1", prompts=t)
        assert w._prompt_template == "A {{subject}}"

    def test_none_means_the_shipped_templates(self):
        from claimlens.workers.extractor import Extractor
        w = Extractor(api_key="k", model="m", base_url="http://fake/v1")
        assert w._prompt_template == settings.PROMPTS["extractor_prompt"]


# Every component that writes a report: module, class, constructor kwargs, child services it builds.
REPORTERS = [
    ("claimlens.pipelines.refchecker", "RefCheckerPipeline",
     dict(extractor_model="e", checker_model="c"), ("ExtractionService", "CheckingService")),
    ("claimlens.pipelines.ragchecker", "RagCheckerPipeline",
     dict(extractor_model="e", checker_model="c"), ("ExtractionService", "CheckingService")),
    ("claimlens.pipelines.faithfulness", "FaithfulnessPipeline",
     dict(extractor_model="e", checker_model="c"), ("ExtractionService", "CheckingService")),
    ("claimlens.pipelines.compare", "ComparePipeline",
     dict(extractor_model="e", checker_model="c"), ("ExtractionService", "CheckingService")),
    ("claimlens.eval.checkereval", "CheckerEvaluator",
     dict(checker_model="c"), ("CheckingService",)),
    ("claimlens.eval.extractoreval", "ExtractorEvaluator",
     dict(extractor_model="e", checker_model="c", atomizer_model="a"),
     ("ExtractionService", "AtomizationService")),
]
IDS = [r[1] for r in REPORTERS]


class TestReportersPassOneTemplateSetDown:
    """No component may build a child with the shipped templates while the
    run uses custom ones — that would silently mix prompt sets."""

    @pytest.mark.parametrize("module, cls, kwargs, children", REPORTERS, ids=IDS)
    def test_every_child_gets_the_same_dict(self, tmp_path, module, cls, kwargs, children):
        templates = _custom(tmp_path)
        patches = [patch(f"{module}.{child}") for child in children]
        mocks = [p.start() for p in patches]
        try:
            component = getattr(importlib.import_module(module), cls)(**kwargs, prompts=templates)
        finally:
            for p in patches:
                p.stop()
        assert component._prompts is templates
        for child, mock in zip(children, mocks):
            assert mock.call_args_list, f"{cls} built no {child}"
            for call in mock.call_args_list:
                assert call.kwargs["prompts"] is templates, f"{cls} → {child}"

    @pytest.mark.parametrize("module, cls", [r[:2] for r in REPORTERS], ids=IDS)
    def test_every_report_meta_carries_the_hash(self, module, cls):
        source = inspect.getsource(importlib.import_module(module))
        calls = source.count("meta = build_meta(")
        assert calls >= 1
        assert source.count("prompts_sha256=settings.prompts_sha256(self._prompts)") == calls


    def test_extractor_eval_checkers_built_at_run_time_get_the_dict_too(self):
        """eval extractor builds its CheckingServices during the run, not in __init__."""
        source = inspect.getsource(importlib.import_module("claimlens.eval.extractoreval"))
        builds = re.findall(r"= CheckingService\((.*?)\n\s*\)", source, re.S)
        assert len(builds) == 2
        assert all("prompts=self._prompts" in args for args in builds)


class TestRecordAndFindings:

    def test_both_documents_carry_the_hash_of_the_templates_used(self, tmp_path):
        from claimlens.pipelines.refchecker import RefCheckerPipeline
        templates = _custom(tmp_path)
        with patch("claimlens.pipelines.refchecker.ExtractionService"), \
             patch("claimlens.pipelines.refchecker.CheckingService"):
            p = RefCheckerPipeline(extractor_model="e", checker_model="c",
                                   verbosity="silent", prompts=templates)

        async def passthrough(data):
            return data
        p._extraction.run = passthrough
        p._checking.run = passthrough
        p.run_sync([{"response": "r", "reference": ["ref"]}])
        assert p.last_report["_meta"]["prompts_sha256"] == prompts_sha256(templates)
        assert p.last_findings["_meta"]["prompts_sha256"] == prompts_sha256(templates)


class TestEntryPoints:

    def _cli(self, tmp_path, monkeypatch, command, target, *extra):
        from typer.testing import CliRunner
        from claimlens.cli import app
        seen = {}

        class _Built:
            def __init__(self, **kwargs):
                seen.update(kwargs)

            def run_sync(self, data):
                return data

        monkeypatch.setattr(target, _Built)
        src = _write(tmp_path, [{"response": "r", "reference": ["x"]}], "d.json")
        models = ["-m", "m"] if command == "extract" else ["-e", "e", "-c", "c"]
        result = CliRunner().invoke(app, [command, str(src), *models, "-o", str(tmp_path / "o.json"), *extra])
        return result, seen

    def test_cli_resolves_the_file_before_building(self, tmp_path, monkeypatch):
        path = _write(tmp_path, {"extractor_prompt": "Mine: {{text}}"})
        result, seen = self._cli(tmp_path, monkeypatch, "extract",
                                 "claimlens.services.extraction.ExtractionService", "--prompts", str(path))
        assert result.exit_code == 0, result.output
        assert seen["prompts"]["extractor_prompt"] == "Mine: {{text}}"

    def test_cli_without_the_flag_passes_the_shipped_templates(self, tmp_path, monkeypatch):
        result, seen = self._cli(tmp_path, monkeypatch, "extract",
                                 "claimlens.services.extraction.ExtractionService")
        assert result.exit_code == 0, result.output
        assert seen["prompts"] == settings.PROMPTS

    @pytest.mark.parametrize("command, target", [
        ("extract", "claimlens.services.extraction.ExtractionService"),
        ("refcheck", "claimlens.pipelines.refchecker.RefCheckerPipeline"),
    ])
    def test_a_bad_file_ends_the_command_before_anything_is_built(self, tmp_path, monkeypatch, command, target):
        path = _write(tmp_path, {"extractor_promt": "typo"})
        result, seen = self._cli(tmp_path, monkeypatch, command, target, "--prompts", str(path))
        assert result.exit_code == 1
        assert not seen

    def test_library_verb_resolves_the_path_and_keeps_it_in_args(self, tmp_path, monkeypatch):
        import claimlens
        seen = {}

        class _Pipeline:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self.last_report = {"_meta": {}, "runs": []}
                self.last_findings = {"_meta": {}, "runs": []}

            def run_sync(self, data):
                return data

        monkeypatch.setattr("claimlens.pipelines.refchecker.RefCheckerPipeline", _Pipeline)
        path = _write(tmp_path, {"extractor_prompt": "Mine: {{text}}"})
        record, _ = claimlens.refcheck([{"response": "r", "reference": ["x"]}],
                                       extractor_model="e", checker_model="c", prompts=path)
        assert seen["prompts"]["extractor_prompt"] == "Mine: {{text}}"
        assert record["_args"]["prompts"] == str(path)
