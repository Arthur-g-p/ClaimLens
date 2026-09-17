"""
The HTML viewer: one self-contained page rendered from the record and the
findings, dispatched on _meta.report_type. The page must embed both
documents unchanged, refuse what it cannot read, load nothing from the
network, carry a metric roster equal to its pipeline's, and be written by
the report commands unless --no-html says otherwise. No browser here: the
JavaScript is not executed, and no generated output is checked in: the
records the tests need are built here.
"""

import json
import re

import pytest
from typer.testing import CliRunner

import claimlens
from claimlens.cli import app
from claimlens.exceptions import ViewerError
from claimlens.utils import REPORT_SCHEMA_VERSION, build_meta
from claimlens.viewer import TEMPLATES, render_html, supported_report_types

TYPES = ["faithcheck", "ragcheck"]
PIPELINES = {
    "ragcheck": ("claimlens.pipelines.ragchecker", "RagCheckerPipeline"),
    "faithcheck": ("claimlens.pipelines.faithfulness", "FaithfulnessPipeline"),
}


def _documents(report_type):
    """A two-run record and its findings, small but schema-shaped, with the
    characters that can break an embedded script: a closing script tag,
    quotes, angle brackets, an ampersand, non-ASCII, a line separator."""
    special = 'Kepler-22b "is" a <super-Earth> & lies 600 ly away \u2014 caf\u00e9\u2028end'
    item = {"query_id": "", "query": "Where is " + special, "response": "</script>" + special,
            "is_abstention": False, "retrieved_context": [{"doc_id": "d1", "text": "<b>" + special}],
            "response_claims": [{"subject": "Kepler-22b", "predicate": "is", "object": "<super-Earth>"}],
            "retrieved2response": [[{"verdict": "Neutral", "explanation": "no </script> mention"}]],
            "metrics": {"faithfulness": None}}
    meta = build_meta(report_type, timestamp="t", duration_seconds=1, total_items=1,
                      evaluated_items=1, dropped_items=0)
    run = {"_meta": meta, "metrics": {"faithfulness": None},
           "counts": {"support": {}, "pipeline": {}, "abstention": {}, "reliability": {}}, "items": [item]}
    args = {"command": report_type, "input_file": "x </script> .json"}
    record = {"_args": args, "_meta": meta, "metrics": {"faithfulness": None},
              "variance": {"faithfulness": {"n": 2, "std": None, "min": None, "max": None, "values": [None, None]}},
              "runs": [run, run]}
    findings = {"_args": args, "_meta": meta,
                "runs": [{"_meta": meta, "findings": {"hallucination": [{"query": item["query"], "claim": "x </script> y"}]}}] * 2}
    return record, findings


def _pipeline(report_type):
    import importlib
    module, name = PIPELINES[report_type]
    return getattr(importlib.import_module(module), name)


def _record(report_type="ragcheck", **meta_overrides):
    meta = build_meta(report_type, timestamp="t", duration_seconds=0,
                      total_items=0, evaluated_items=0, dropped_items=0)
    meta.update(meta_overrides)
    return {"_args": {"command": report_type}, "_meta": meta,
            "metrics": {}, "variance": {}, "runs": []}


def _embedded(page, element_id):
    m = re.search(r'<script type="application/json" id="%s">(.*?)</script>' % element_id, page, re.S)
    assert m, f"no embedded document {element_id!r}"
    return m.group(1)


class TestRender:

    @pytest.mark.parametrize("report_type", TYPES)
    def test_round_trips_both_documents(self, report_type):
        record, findings = _documents(report_type)
        page = render_html(record, findings)
        assert json.loads(_embedded(page, "record")) == record
        assert json.loads(_embedded(page, "findings")) == findings
        for marker in ("__RECORD__", "__FINDINGS__", "/*COMMON_CSS*/", "/*COMMON_JS*/"):
            assert marker not in page

    def test_angle_bracket_in_a_claim_cannot_close_the_script_tag(self):
        record = _record()
        record["runs"] = [{"items": [{"response": "</script><b>x</b>", "query": "<q>"}]}]
        body = _embedded(render_html(record), "record")
        assert "<" not in body
        assert json.loads(body) == record

    def test_missing_findings_embeds_an_empty_document(self):
        assert json.loads(_embedded(render_html(_record()), "findings")) == {}

    def test_unknown_report_type_names_the_viewable_ones(self):
        with pytest.raises(ViewerError, match=r"'refcheck'.*faithcheck, ragcheck"):
            render_html(_record("refcheck"))

    def test_schema_mismatch_refuses(self):
        with pytest.raises(ViewerError, match="schema_version"):
            render_html(_record(schema_version=REPORT_SCHEMA_VERSION - 1))

    def test_viewable_report_types(self):
        assert supported_report_types() == TYPES

    def test_every_template_declares_the_schema_it_reads(self):
        for report_type in supported_report_types():
            source = (TEMPLATES / f"{report_type}.html").read_text(encoding="utf-8")
            m = re.search(r'<meta name="claimlens-schema" content="(\d+)">', source)
            assert m and int(m.group(1)) == REPORT_SCHEMA_VERSION, report_type

    @pytest.mark.parametrize("report_type", TYPES)
    def test_loads_nothing_from_the_network(self, report_type):
        page = render_html(_record(report_type))
        assert not re.search(r'(src|href)\s*=\s*"https?://', page)
        assert "@import" not in page


class TestFacade:

    def test_render_html_is_public(self):
        assert "render_html" in claimlens.__all__
        assert claimlens.render_html is render_html


class _FakePipeline:
    """Constructed by a report command; produces a schema-valid record."""
    report_type = "ragcheck"

    def __init__(self, **kwargs):
        self.last_report = {k: v for k, v in _record(self.report_type).items() if k != "_args"}
        self.last_findings = {"_meta": self.last_report["_meta"], "runs": []}

    def run_sync(self, data):
        return data


@pytest.fixture(params=TYPES)
def report_cli(request, monkeypatch, tmp_path):
    report_type = request.param
    fake = type("Fake", (_FakePipeline,), {"report_type": report_type})
    monkeypatch.setattr(".".join(PIPELINES[report_type]), fake)
    monkeypatch.setenv("EXTRACTOR_API_KEY", "x")
    monkeypatch.setenv("CHECKER_API_KEY", "x")
    src = tmp_path / "data.json"
    src.write_text(json.dumps([{"response": "r", "gt_answer": "g", "retrieved_context": ["c"]}]), encoding="utf-8")
    out = tmp_path / "out" / f"data_{report_type}.json"

    def invoke(*extra):
        result = CliRunner().invoke(app, [report_type, str(src), "-e", "e", "-c", "c", "-o", str(out), *extra])
        assert result.exit_code == 0, result.output
        return out

    return invoke


class TestCli:

    def test_html_is_written_by_default(self, report_cli):
        out = report_cli()
        page = out.with_suffix(".html")
        assert page.exists()
        record = json.loads(out.read_text(encoding="utf-8"))
        assert json.loads(_embedded(page.read_text(encoding="utf-8"), "record")) == record
        assert record["_args"]["html"] is True
        assert "html" not in record["_args"]["_explicit"]

    def test_no_html_skips_the_page_and_is_recorded(self, report_cli):
        out = report_cli("--no-html")
        assert not out.with_suffix(".html").exists()
        assert out.exists() and out.with_name(out.stem + "_findings.json").exists()
        record = json.loads(out.read_text(encoding="utf-8"))
        assert record["_args"]["html"] is False
        assert "html" in record["_args"]["_explicit"]


class TestRosterParity:
    """Each template carries its own copy of its pipeline's metric roster;
    the record does not. This pins every copy to its source."""

    @staticmethod
    def _roster(report_type):
        source = (TEMPLATES / f"{report_type}.html").read_text(encoding="utf-8")
        m = re.search(r'<script type="application/json" id="roster">(.*?)</script>', source, re.S)
        assert m, f"{report_type}.html has no roster island"
        return json.loads(m.group(1))

    @pytest.mark.parametrize("report_type", TYPES)
    def test_groups_behavior_health_and_directions_match_the_pipeline(self, report_type):
        P = _pipeline(report_type)
        roster = self._roster(report_type)
        assert [(g[0], g[2]) for g in roster["groups"]] == [(n, list(k)) for n, k in P._VARIANCE_SECTIONS["metrics"]]
        assert roster["behavior"] == list(P._VARIANCE_SECTIONS["behavior"])
        assert roster["health"] == list(P._VARIANCE_SECTIONS["health"])
        assert roster["directions"] == P._METRIC_DIRECTIONS

    @pytest.mark.parametrize("report_type", TYPES)
    def test_universes_name_real_abstention_counts(self, report_type):
        # The keys counts.abstention carries come from these two functions.
        from claimlens.pipelines import faithfulness, ragchecker
        ab = ragchecker._abstention_breakdown([]) if report_type == "ragcheck" else faithfulness.abstention_counts([])
        roster = self._roster(report_type)
        named = {k for _, _, keys in roster["groups"] for k in keys} | set(roster["behavior"])
        for metric, fields in roster["universes"].items():
            assert metric in named, metric
            for field in fields:
                assert field in ab, (metric, field)
