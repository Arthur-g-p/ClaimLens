"""
The HTML viewer: one self-contained page rendered from the record and the
findings, dispatched on _meta.report_type. The page must embed both
documents unchanged, refuse what it cannot read, load nothing from the
network, and be written by the ragcheck command unless --no-html says
otherwise. No browser here: the JavaScript is not executed.
"""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import claimlens
from claimlens.cli import app
from claimlens.exceptions import ViewerError
from claimlens.utils import REPORT_SCHEMA_VERSION, build_meta
from claimlens.viewer import TEMPLATES, render_html, supported_report_types

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/ragcheck/results/kepler22b_ragcheck_2.json"


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

    def test_example_round_trips_both_documents(self):
        record = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        findings = json.loads(EXAMPLE.with_name(EXAMPLE.stem + "_findings.json").read_text(encoding="utf-8"))
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
        with pytest.raises(ViewerError, match=r"'faithcheck'.*ragcheck"):
            render_html(_record("faithcheck"))

    def test_schema_mismatch_refuses(self):
        with pytest.raises(ViewerError, match="schema_version"):
            render_html(_record(schema_version=REPORT_SCHEMA_VERSION - 1))

    def test_only_ragcheck_for_now(self):
        assert supported_report_types() == ["ragcheck"]

    def test_every_template_declares_the_schema_it_reads(self):
        for report_type in supported_report_types():
            source = (TEMPLATES / f"{report_type}.html").read_text(encoding="utf-8")
            m = re.search(r'<meta name="claimlens-schema" content="(\d+)">', source)
            assert m and int(m.group(1)) == REPORT_SCHEMA_VERSION, report_type

    def test_loads_nothing_from_the_network(self):
        page = render_html(_record())
        assert not re.search(r'(src|href)\s*=\s*"https?://', page)
        assert "@import" not in page


class TestFacade:

    def test_render_html_is_public(self):
        assert "render_html" in claimlens.__all__
        assert claimlens.render_html is render_html


class _FakePipeline:
    """Constructed by the ragcheck command; produces a schema-valid record."""

    def __init__(self, **kwargs):
        self.last_report = {k: v for k, v in _record().items() if k != "_args"}
        self.last_findings = {"_meta": self.last_report["_meta"], "runs": []}

    def run_sync(self, data):
        return data


@pytest.fixture
def ragcheck_cli(monkeypatch, tmp_path):
    monkeypatch.setattr("claimlens.pipelines.ragchecker.RagCheckerPipeline", _FakePipeline)
    monkeypatch.setenv("EXTRACTOR_API_KEY", "x")
    monkeypatch.setenv("CHECKER_API_KEY", "x")
    src = tmp_path / "data.json"
    src.write_text(json.dumps([{"response": "r", "gt_answer": "g", "retrieved_context": ["c"]}]), encoding="utf-8")
    out = tmp_path / "out" / "data_ragcheck.json"

    def invoke(*extra):
        result = CliRunner().invoke(app, ["ragcheck", str(src), "-e", "e", "-c", "c", "-o", str(out), *extra])
        assert result.exit_code == 0, result.output
        return out

    return invoke


class TestCli:

    def test_html_is_written_by_default(self, ragcheck_cli):
        out = ragcheck_cli()
        page = out.with_suffix(".html")
        assert page.exists()
        record = json.loads(out.read_text(encoding="utf-8"))
        assert json.loads(_embedded(page.read_text(encoding="utf-8"), "record")) == record
        assert record["_args"]["html"] is True
        assert "html" not in record["_args"]["_explicit"]

    def test_no_html_skips_the_page_and_is_recorded(self, ragcheck_cli):
        out = ragcheck_cli("--no-html")
        assert not out.with_suffix(".html").exists()
        assert out.exists() and out.with_name(out.stem + "_findings.json").exists()
        record = json.loads(out.read_text(encoding="utf-8"))
        assert record["_args"]["html"] is False
        assert "html" in record["_args"]["_explicit"]


class TestRosterParity:
    """The template carries its own copy of the pipeline's metric roster;
    the record does not. This pins the copy to the source."""

    @staticmethod
    def _roster():
        source = (TEMPLATES / "ragcheck.html").read_text(encoding="utf-8")
        m = re.search(r'<script type="application/json" id="roster">(.*?)</script>', source, re.S)
        assert m, "ragcheck.html has no roster island"
        return json.loads(m.group(1))

    def test_groups_directions_and_health_match_the_pipeline(self):
        from claimlens.pipelines.ragchecker import RagCheckerPipeline as P
        roster = self._roster()
        assert [(g[0], g[2]) for g in roster["groups"]] == [(n, list(k)) for n, k in P._VARIANCE_SECTIONS["metrics"]]
        assert roster["health"] == list(P._VARIANCE_SECTIONS["health"])
        assert roster["directions"] == P._METRIC_DIRECTIONS

    def test_universes_name_real_abstention_counts(self):
        roster = self._roster()
        record = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        ab = record["runs"][0]["counts"]["abstention"]
        named = {k for _, _, keys in roster["groups"] for k in keys}
        for metric, (num, den) in roster["universes"].items():
            assert metric in named
            assert num in ab and den in ab, metric
