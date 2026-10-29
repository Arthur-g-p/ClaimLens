"""The prompt map, --prompts, and the request built from a prompt.

Every prompt is {system, instructions, input}. --prompts lays a user file over
the shipped prompt_map.json (only the keys it changes). The entry point (CLI or
library verb) resolves the path once — fail early: unknown key, broken JSON,
wrong shape, a placeholder where none belongs — and everything below receives
the map as ``templates``, passed down like --concurrency. Its hash lands in
_meta so a record still names the prompts that produced it after the file is
edited; the path itself is in _args.

A prompt is sent as system message + one user message of two blocks
(instructions, filled input); LLMClient marks the instructions block or joins
the two back into the one string the prompt always made.
"""

import importlib
import inspect
import json
import re
from unittest.mock import patch

import pytest

from claimlens import prompts
from claimlens.llmclient import _as_strings, _with_cache_marker
from claimlens.prompts import describe_prompts, format_prompt, prompt_messages, prompts_sha256, resolve_prompts


def _write(tmp_path, data, name="mine.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


def _prompt(instructions, input, system="Role."):
    return {"system": system, "instructions": instructions, "input": input}


MINE = {"extractor_prompt": _prompt("Mine: ", "{{text}}", system="My role.")}


def _custom(tmp_path):
    return resolve_prompts(_write(tmp_path, MINE))


class TestResolve:

    def test_default_is_the_shipped_file(self):
        assert resolve_prompts() is prompts.PROMPTS

    def test_every_shipped_prompt_has_the_three_parts(self):
        for key, prompt in prompts.PROMPTS.items():
            assert tuple(prompt) == prompts.PARTS, key

    def test_file_replaces_only_its_keys(self, tmp_path):
        p = _custom(tmp_path)
        assert p["extractor_prompt"] == MINE["extractor_prompt"]
        assert p["checker_prompt"] == prompts.PROMPTS["checker_prompt"]
        assert set(p) == set(prompts.PROMPTS)

    def test_an_empty_file_is_the_shipped_set(self, tmp_path):
        p = resolve_prompts(_write(tmp_path, {}))
        assert p == prompts.PROMPTS
        assert prompts_sha256(p) == prompts_sha256(prompts.PROMPTS)

    def test_hash_follows_the_content_not_the_path(self, tmp_path):
        a = resolve_prompts(_write(tmp_path, {"extractor_prompt": _prompt("x", "{{text}}")}, "a.json"))
        b = resolve_prompts(_write(tmp_path, {"extractor_prompt": _prompt("x", "{{text}}")}, "b.json"))
        c = resolve_prompts(_write(tmp_path, {"extractor_prompt": _prompt("y", "{{text}}")}, "c.json"))
        assert prompts_sha256(a) == prompts_sha256(b) != prompts_sha256(c)
        assert prompts_sha256(a) != prompts_sha256(resolve_prompts())
        assert len(prompts_sha256(a)) == 12

    def test_the_system_message_is_part_of_the_hash(self, tmp_path):
        a = resolve_prompts(_write(tmp_path, {"extractor_prompt": _prompt("x", "{{text}}", "One.")}, "a.json"))
        b = resolve_prompts(_write(tmp_path, {"extractor_prompt": _prompt("x", "{{text}}", "Two.")}, "b.json"))
        assert prompts_sha256(a) != prompts_sha256(b)

    def test_config_line_says_shipped_or_custom_with_the_hash(self, tmp_path):
        shipped, custom = resolve_prompts(), _custom(tmp_path)
        assert describe_prompts(shipped) == f"{prompts.PROMPT_PATH} · sha256 {prompts_sha256(shipped)}"
        assert describe_prompts(custom) == (f"custom extractor_prompt, over prompt_map.json · "
                                            f"sha256 {prompts_sha256(custom)}")

    @pytest.mark.parametrize("content, message", [
        ({"extractor_promt": _prompt("x", "{{text}}")}, "unknown prompt key(s) extractor_promt"),
        ({"extractor_prompt": "the old one-string shape {{text}}"}, '"extractor_prompt" must be {'),
        ({"extractor_prompt": {"instructions": "x", "input": "{{text}}"}}, '"extractor_prompt" must be {'),
        ({"extractor_prompt": _prompt("x", 3)}, '"extractor_prompt" must be {'),
        ({"extractor_prompt": _prompt("x", "{{text}}", system="About {{text}}")}, "placeholder in its system message"),
        ({"extractor_prompt": _prompt("Text: {{text}}", "")}, "item placeholder in its instructions"),
        ({"checker_prompt_joint": _prompt("x", "{{claims}}")}, "input lacks {{reference}}"),
        ({"extractor_prompt": _prompt("x", "{{text}} {{txt}}")}, "placeholder(s) nothing fills: {{txt}}"),
        (["not", "an", "object"], "expected a JSON object"),
        ("{broken", "not valid JSON"),
    ])
    def test_bad_files_fail_before_any_request(self, tmp_path, content, message):
        with pytest.raises(ValueError, match=re.escape(message)):
            resolve_prompts(_write(tmp_path, content))

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="Prompt file not found"):
            resolve_prompts(tmp_path / "nope.json")

    def test_the_shipped_map_holds_every_required_key(self):
        assert set(prompts.REQUIRED_KEYS) <= set(prompts.PROMPTS)

    def test_a_shipped_map_missing_a_required_key_fails_at_load(self, tmp_path, monkeypatch):
        shipped = {k: v for k, v in prompts.PROMPTS.items() if k != "checker_prompt_plain"}
        monkeypatch.setattr(prompts, "PROMPT_PATH", _write(tmp_path, shipped, "prompt_map.json"))
        with pytest.raises(ValueError, match="lacks required prompt.s.: checker_prompt_plain"):
            prompts._load_shipped()

    def test_overridden_keys_are_the_ones_that_differ(self, tmp_path):
        same = {"checker_prompt": prompts.PROMPTS["checker_prompt"]}
        assert prompts.prompts_overridden(resolve_prompts(_write(tmp_path, same))) == []
        assert prompts.prompts_overridden(_custom(tmp_path)) == ["extractor_prompt"]
        assert prompts.prompts_overridden(prompts.PROMPTS) == []

    def test_a_map_without_a_key_names_it(self):
        with pytest.raises(ValueError, match='the prompt map has no "checker_prompt_joint_ctx"'):
            prompts.prompt(prompts.PROMPTS, "checker_prompt_joint_ctx")

    def test_a_dict_is_refused_clearly(self):
        with pytest.raises(ValueError, match="takes the path of a prompt file"):
            resolve_prompts({"extractor_prompt": _prompt("x", "{{text}}")})


class TestSplitTemplate:
    """Converting a one-string template (an older prompt map, a prompt built as one text)."""

    def test_cut_at_the_first_item_placeholder_schema_does_not_count(self):
        t = prompts.split_template("Rules {{schema}} more. Text: {{text}} end", "Role.")
        assert t == {"system": "Role.", "instructions": "Rules {{schema}} more. Text: ", "input": "{{text}} end"}

    @pytest.mark.parametrize("key", sorted(prompts.PROMPTS))
    def test_every_shipped_prompt_round_trips(self, key):
        p = prompts.PROMPTS[key]
        assert prompts.split_template(p["instructions"] + p["input"], p["system"]) == p


class TestPromptMessages:
    """The request built from a prompt, and its contract with the client's marker."""

    VALUES = {"text": "T", "claim": "C", "claims": "[1] C", "reference": "R",
              "subject": "S", "predicate": "P", "object": "O", "response": "X"}

    def test_system_then_one_user_message_of_two_blocks(self):
        msgs = prompt_messages(_prompt("Rules. ", "Text: {{text}}", system="My role."), {"text": "Goethe"})
        assert msgs == [{"role": "system", "content": "My role."},
                        {"role": "user", "content": [{"type": "text", "text": "Rules. "},
                                                     {"type": "text", "text": "Text: Goethe"}]}]

    @pytest.mark.parametrize("key", sorted(prompts.PROMPTS))
    def test_joined_blocks_are_the_old_one_string_request(self, key):
        prompt = prompts.PROMPTS[key]
        joined = _as_strings(prompt_messages(prompt, self.VALUES))
        assert joined == [{"role": "system", "content": prompt["system"]},
                          {"role": "user", "content": format_prompt(prompt["instructions"] + prompt["input"],
                                                                    self.VALUES)}]

    @pytest.mark.parametrize("key", sorted(prompts.PROMPTS))
    def test_the_client_marks_exactly_the_instructions_block(self, key):
        """prompt_messages and llmclient._with_cache_marker are counterparts."""
        prompt = prompts.PROMPTS[key]
        marked, chars = _with_cache_marker(prompt_messages(prompt, self.VALUES))
        instructions, item = marked[1]["content"]
        assert instructions == {"type": "text", "text": prompt["instructions"], "cache_control": {"type": "ephemeral"}}
        assert "cache_control" not in item
        assert marked[0] == {"role": "system", "content": prompt["system"]}
        assert chars == len(prompt["system"]) + len(prompt["instructions"])   # the whole cached prefix


class TestWorkersUseTheTemplatesTheyWereGiven:

    def test_extractor(self, tmp_path):
        from claimlens.workers.extractor import Extractor
        w = Extractor(api_key="k", model="m", base_url="http://fake/v1", templates=_custom(tmp_path))
        system, user = w._build_messages("Goethe.")
        assert system["content"] == "My role."
        assert user["content"] == [{"type": "text", "text": "Mine: "}, {"type": "text", "text": "Goethe."}]

    def test_checker_joint(self, tmp_path):
        from claimlens.workers.checker import Checker
        t = resolve_prompts(_write(tmp_path, {"checker_prompt_joint": _prompt("J ", "{{claims}} | {{reference}}")}))
        w = Checker(api_key="k", model="m", base_url="http://fake/v1", templates=t)
        instructions, item = w._build_joint_messages([(1, "c")], ["r"])[-1]["content"]
        assert instructions["text"] == "J "
        assert item["text"].startswith("[1] c | ")

    def test_atomizer(self, tmp_path):
        from claimlens.workers.atomizer import Atomizer
        t = resolve_prompts(_write(tmp_path, {"atomizer_prompt": _prompt(
            "A ", "{{subject}} {{predicate}} {{object}} {{response}}")}))
        w = Atomizer(api_key="k", model="m", base_url="http://fake/v1", templates=t)
        assert w._prompt_template == t["atomizer_prompt"]

    def test_none_means_the_shipped_templates(self):
        from claimlens.workers.extractor import Extractor
        w = Extractor(api_key="k", model="m", base_url="http://fake/v1")
        assert w._prompt_template == prompts.PROMPTS["extractor_prompt"]


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
    def test_every_child_gets_the_same_map(self, tmp_path, module, cls, kwargs, children):
        templates = _custom(tmp_path)
        patches = [patch(f"{module}.{child}") for child in children]
        mocks = [p.start() for p in patches]
        try:
            component = getattr(importlib.import_module(module), cls)(**kwargs, templates=templates)
        finally:
            for p in patches:
                p.stop()
        assert component._templates is templates
        for child, mock in zip(children, mocks):
            assert mock.call_args_list, f"{cls} built no {child}"
            for call in mock.call_args_list:
                assert call.kwargs["templates"] is templates, f"{cls} → {child}"

    @pytest.mark.parametrize("module, cls", [r[:2] for r in REPORTERS], ids=IDS)
    def test_every_report_meta_carries_hash_and_marker_state(self, module, cls):
        source = inspect.getsource(importlib.import_module(module))
        calls = source.count("meta = build_meta(")
        assert calls >= 1
        assert source.count("prompts_sha256=prompts.prompts_sha256(self._templates)") == calls
        assert source.count("prompt_cache_marker=GLOBAL_STATS.cache_markers()") == calls
        assert source.count("prompts_overridden=prompts.prompts_overridden(self._templates)") == calls

    def test_extractor_eval_checkers_built_at_run_time_get_the_map_too(self):
        """eval extractor builds its CheckingServices during the run, not in __init__."""
        source = inspect.getsource(importlib.import_module("claimlens.eval.extractoreval"))
        builds = re.findall(r"= CheckingService\((.*?)\n\s*\)", source, re.S)
        assert len(builds) == 2
        assert all("templates=self._templates" in args for args in builds)


class TestRecordAndFindings:

    def test_both_documents_carry_the_hash_and_the_marker_state(self, tmp_path):
        from claimlens.pipelines.refchecker import RefCheckerPipeline
        from claimlens.stats import GLOBAL_STATS
        templates = _custom(tmp_path)
        with patch("claimlens.pipelines.refchecker.ExtractionService"), \
             patch("claimlens.pipelines.refchecker.CheckingService"):
            p = RefCheckerPipeline(extractor_model="e", checker_model="c",
                                   verbosity="silent", templates=templates)

        async def passthrough(data):
            GLOBAL_STATS.record_cache_marker("e", "http://x/v1", "sent")
            return data
        p._extraction.run = passthrough
        p._checking.run = passthrough
        p.run_sync([{"response": "r", "reference": ["ref"]}])
        for doc in (p.last_report, p.last_findings):
            assert doc["_meta"]["prompts_sha256"] == prompts_sha256(templates)
            assert doc["_meta"]["prompt_cache_marker"]["e"] == "sent"
            assert doc["_meta"]["prompts_overridden"] == ["extractor_prompt"]


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
        path = _write(tmp_path, MINE)
        result, seen = self._cli(tmp_path, monkeypatch, "extract",
                                 "claimlens.services.extraction.ExtractionService", "--prompts", str(path))
        assert result.exit_code == 0, result.output
        assert seen["templates"]["extractor_prompt"] == MINE["extractor_prompt"]

    def test_cli_without_the_flag_passes_the_shipped_templates(self, tmp_path, monkeypatch):
        result, seen = self._cli(tmp_path, monkeypatch, "extract",
                                 "claimlens.services.extraction.ExtractionService")
        assert result.exit_code == 0, result.output
        assert seen["templates"] == prompts.PROMPTS

    @pytest.mark.parametrize("command, target", [
        ("extract", "claimlens.services.extraction.ExtractionService"),
        ("refcheck", "claimlens.pipelines.refchecker.RefCheckerPipeline"),
    ])
    def test_a_bad_file_ends_the_command_before_anything_is_built(self, tmp_path, monkeypatch, command, target):
        path = _write(tmp_path, {"extractor_promt": _prompt("x", "{{text}}")})
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
        path = _write(tmp_path, MINE)
        record, _ = claimlens.refcheck([{"response": "r", "reference": ["x"]}],
                                       extractor_model="e", checker_model="c", prompts=path)
        assert seen["templates"]["extractor_prompt"] == MINE["extractor_prompt"]
        assert "prompts" not in seen
        assert record["_args"]["prompts"] == str(path)
