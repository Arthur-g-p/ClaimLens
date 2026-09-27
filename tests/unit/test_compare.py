"""
Unit tests for ComparePipeline: validation, direction wiring, the report
projection, the neutral words, and the claim that ragcheck's precision
and recall are compare with the GT answer as ground truth. Workers patched
out at construction — no LLM.
"""

import logging

import pytest
from unittest.mock import patch

from claimlens.exceptions import InvalidInputError
from claimlens.pipelines.compare import ComparePipeline, METRIC_NAMES
from claimlens.pipelines.ragchecker import RagCheckerPipeline


EXT = "ext-model"
CHK = "chk-model"

A_KG = f"{EXT}_a_kg"
B_KG = f"{EXT}_b_kg"


def _pipeline(**kwargs):
    with patch("claimlens.services.extraction.Extractor"), \
         patch("claimlens.services.checking.Checker"):
        return ComparePipeline(extractor_model=EXT, checker_model=CHK, **kwargs)


@pytest.fixture
def pipeline():
    return _pipeline()


def _triplet(obj, **verdicts):
    """A claim with flat verdicts: _triplet("x", b2a="Entailment")."""
    t = {"subject": "Nile", "predicate": "is", "object": obj}
    for direction, verdict in verdicts.items():
        t[f"{CHK}_{direction}_verdict"] = verdict
        t[f"{CHK}_{direction}_explanation"] = f"{direction} says {verdict}"
    return t


def _checked_item(**overrides):
    """An item as it looks after both extractions and both directions:
    A's claims judged against B's text (b2a), B's against A's (a2b)."""
    item = {
        "id": "nile",
        "a": "The Nile is the longest river. It flows north. It is 6,650 km long.",
        "b": "The Nile, the longest river, is 7,000 km long.",
        A_KG: [_triplet("longest river", b2a="Entailment"),
               _triplet("flowing north", b2a="Neutral"),
               _triplet("6,650 km long", b2a="Contradiction")],
        B_KG: [_triplet("longest river", a2b="Entailment"),
               _triplet("7,000 km long", a2b="Contradiction")],
    }
    item.update(overrides)
    return item


# ── Construction ─────────────────────────────────────────────────────────────

class TestConstruction:

    def test_directions_check_each_side_against_the_other_text(self, pipeline):
        wiring = {d.name: (d.kg_key, d.reference_key) for d, _ in pipeline._directions}
        assert wiring == {"b2a": (A_KG, "b"), "a2b": (B_KG, "a")}

    def test_namespaces_follow_the_direction(self, pipeline):
        keys = {d.name: s.verdict_key for d, s in pipeline._directions}
        assert keys == {"b2a": f"{CHK}_b2a_verdict", "a2b": f"{CHK}_a2b_verdict"}

    def test_neither_extraction_marks_abstention(self, pipeline):
        """Neither text is a response: an empty extraction is a text that
        states nothing, not a refusal."""
        assert all(not s._mark_abstention for s in pipeline._extract.values())


# ── Validation ───────────────────────────────────────────────────────────────

class TestValidate:

    def test_both_texts_required(self, pipeline):
        valid = pipeline._validate([{"a": "x", "b": "y"}, {"a": "x"}, {"b": "y"}])
        assert len(valid) == 1

    def test_empty_text_is_data_not_missing(self, pipeline):
        assert len(pipeline._validate([{"a": "x", "b": ""}])) == 1

    def test_null_text_is_missing(self, pipeline):
        assert len(pipeline._validate([{"a": "x", "b": None}, {"a": "x", "b": "y"}])) == 1

    def test_non_objects_skipped(self, pipeline):
        assert len(pipeline._validate(["text", {"a": "x", "b": "y"}])) == 1

    def test_nothing_valid_raises(self, pipeline):
        with pytest.raises(InvalidInputError):
            pipeline._validate([{"a": "only one text"}])


# ── Known verdicts ───────────────────────────────────────────────────────────

class TestPrefill:

    def test_claims_against_an_empty_text_are_neutral_without_a_request(self, pipeline):
        item = {"a": "The Nile is long.", "b": "  ",
                A_KG: [{"subject": "Nile", "predicate": "is", "object": "long"}], B_KG: []}
        pipeline._prefill_known_verdicts([item])
        assert item[A_KG][0][f"{CHK}_b2a_verdict"] == "Neutral"

    def test_non_empty_texts_are_left_to_the_checker(self, pipeline):
        item = _checked_item()
        for t in item[A_KG]:
            t.pop(f"{CHK}_b2a_verdict")
        pipeline._prefill_known_verdicts([item])
        assert all(f"{CHK}_b2a_verdict" not in t for t in item[A_KG])


# ── Report projection + metrics ──────────────────────────────────────────────

class TestBuildRun:

    def test_skeleton_and_meta(self, pipeline):
        run = pipeline._build_run([_checked_item(), {"a": "no b"}])
        assert list(run) == ["_meta", "metrics", "counts", "items"]
        assert run["_meta"]["report_type"] == "compare"
        assert (run["_meta"]["evaluated_items"], run["_meta"]["dropped_items"]) == (1, 1)

    def test_entry_shape(self, pipeline):
        entry = pipeline._build_run([_checked_item()])["items"][0]
        assert list(entry) == ["id", "a", "b", "a_claims", "b_claims", "b2a", "a2b", "metrics"]
        assert entry["a_claims"][0] == {"subject": "Nile", "predicate": "is", "object": "longest river"}
        # Each verdict array is parallel to the claims it judges.
        assert len(entry["b2a"]) == len(entry["a_claims"]) == 3
        assert len(entry["a2b"]) == len(entry["b_claims"]) == 2
        assert entry["b2a"][2] == {"verdict": "Contradiction", "explanation": "b2a says Contradiction"}

    def test_item_metrics(self, pipeline):
        entry = pipeline._build_run([_checked_item()])["items"][0]
        assert entry["metrics"] == {"a_in_b": 0.3333, "a_contradicted": 0.3333,
                                    "b_in_a": 0.5, "b_contradicted": 0.5}

    def test_id_falls_back_to_the_position(self, pipeline):
        item = _checked_item()
        del item["id"]
        assert pipeline._build_run([{"a": "x"}, item])["items"][0]["id"] == "item-1"

    def test_unjudged_claims_leave_both_sides_of_the_ratio(self, pipeline):
        item = _checked_item()
        item[A_KG][1][f"{CHK}_b2a_verdict"] = None
        item[A_KG][1][f"{CHK}_b2a_error"] = "parse_failure"
        entry = pipeline._build_run([item])["items"][0]
        assert entry["b2a"][1] == {"verdict": None, "explanation": "b2a says Neutral",
                                   "error": "parse_failure"}
        assert entry["metrics"]["a_in_b"] == 0.5          # 1 of 2 judged

    def test_extraction_error_excludes_the_item(self, pipeline):
        item = _checked_item(**{A_KG: [], f"{EXT}_a_extraction_error": "timeout"})
        entry = pipeline._build_run([item])["items"][0]
        assert entry["extraction_errors"] == {"a": "timeout"}
        assert entry["metrics"] == dict.fromkeys(METRIC_NAMES)

    def test_a_text_without_claims_is_not_computable(self, pipeline):
        item = _checked_item(**{B_KG: []})
        entry = pipeline._build_run([item])["items"][0]
        assert entry["metrics"]["b_in_a"] is None
        assert entry["metrics"]["a_in_b"] == 0.3333

    def test_run_metrics_are_the_roster(self, pipeline):
        errored = _checked_item(id="bad", **{B_KG: [], f"{EXT}_b_extraction_error": "parse_failure"})
        run = pipeline._build_run([_checked_item(), errored])
        assert run["metrics"] == {
            "a_in_b": 0.3333, "a_contradicted": 0.3333, "b_in_a": 0.5, "b_contradicted": 0.5,
            "extraction_error_rate": 0.5, "checker_failure_rate": 0.0}

    def test_counts(self, pipeline):
        errored = _checked_item(id="bad", **{B_KG: [], f"{EXT}_b_extraction_error": "parse_failure"})
        counts = pipeline._build_run([_checked_item(), errored])["counts"]
        assert list(counts) == ["support", "pipeline", "reliability"]
        assert counts["support"] == dict.fromkeys(METRIC_NAMES, 1)
        assert counts["reliability"] == {
            "extraction": {"failed": 1, "items": 2, "by_cause": {"a": 0, "b": 1}},
            "checking": {"unjudged": 0, "issued": 5}}
        assert counts["pipeline"]["b2a"]["verdicts"] == 6
        assert counts["pipeline"]["a2b"]["Contradiction"] == 1

    def test_findings_open_the_claim_outcomes(self, pipeline):
        item = _checked_item()
        item[B_KG][0][f"{CHK}_a2b_verdict"] = None
        run = pipeline._build_run([item])
        findings = pipeline._build_findings(run["items"])
        assert list(findings) == ["a_not_in_b", "a_contradicted", "b_not_in_a",
                                  "b_contradicted", "unjudged", "extraction_failed"]
        assert findings["a_not_in_b"] == [{"id": "nile", "claim": "Nile is flowing north",
                                           "explanation": "b2a says Neutral"}]
        assert findings["a_contradicted"][0]["claim"] == "Nile is 6,650 km long"
        assert findings["b_contradicted"][0]["claim"] == "Nile is 7,000 km long"
        assert findings["unjudged"] == [{"id": "nile", "claim": "Nile is longest river",
                                         "side": "b", "cause": "checker_failure"}]
        assert findings["b_not_in_a"] == [] and findings["extraction_failed"] == []

    def test_findings_name_the_failed_side(self, pipeline):
        item = _checked_item(**{A_KG: [], f"{EXT}_a_extraction_error": "timeout"})
        findings = pipeline._build_findings(pipeline._build_run([item])["items"])
        assert findings["extraction_failed"] == [{"id": "nile", "side": "a", "cause": "timeout"}]
        assert findings["b_contradicted"] == []    # the whole item is out


# ── The words: neutral, the reading is the reader's ─────────────────────────

class TestWords:

    def test_labels_blame_nobody(self, pipeline):
        """Which text is right is the reader's call, made in the HTML report;
        the console says what was measured and nothing more."""
        assert pipeline._VARIANCE_LABELS == {
            "a_in_b": "A in B", "a_contradicted": "A conflicts with B",
            "b_in_a": "B in A", "b_contradicted": "B conflicts with A"}

    def test_overlap_has_no_direction_of_its_own(self, pipeline):
        assert "a_in_b" not in pipeline._METRIC_DIRECTIONS
        assert "b_in_a" not in pipeline._METRIC_DIRECTIONS

    def test_variance_groups_match_the_metrics_tree(self, caplog, pipeline):
        pipeline.last_run = pipeline._build_run([_checked_item()])
        with caplog.at_level(logging.INFO):
            pipeline._log_metrics()
        for group, keys in pipeline._VARIANCE_SECTIONS["metrics"]:
            assert f"{group} — its claims" in caplog.text
            for key in keys:
                assert f"{pipeline._VARIANCE_LABELS[key]}:" in caplog.text
        assert "A — its claims checked against the text of B" in caplog.text


# ── ragcheck's generator half is compare ─────────────────────────────────────

class TestRagcheckEquivalence:
    """compare(gt_answer, response) with a as the ground truth gives exactly
    ragcheck's precision and recall from the same verdicts."""

    VERDICTS_RESPONSE = ["Entailment", "Neutral", None, "Contradiction"]   # vs the GT answer
    VERDICTS_GT = ["Entailment", "Entailment", "Neutral"]                  # vs the response

    def _ragcheck_entry(self):
        with patch("claimlens.services.extraction.Extractor"), \
             patch("claimlens.services.checking.Checker"):
            rag = RagCheckerPipeline(extractor_model=EXT, checker_model=CHK)
        item = {
            "query_id": "q", "response": "r", "gt_answer": "g",
            "retrieved_context": [{"doc_id": "000", "text": "c"}],
            f"{EXT}_response_kg": [_triplet(f"r{i}", answer2response=v)
                                   for i, v in enumerate(self.VERDICTS_RESPONSE)],
            f"{EXT}_gt_answer_kg": [_triplet(f"g{i}", response2answer=v)
                                    for i, v in enumerate(self.VERDICTS_GT)],
        }
        return rag._build_run([item])["items"][0]

    def _compare_entry(self):
        item = {
            "id": "q", "a": "g", "b": "r",
            A_KG: [_triplet(f"g{i}", b2a=v) for i, v in enumerate(self.VERDICTS_GT)],
            B_KG: [_triplet(f"r{i}", a2b=v) for i, v in enumerate(self.VERDICTS_RESPONSE)],
        }
        return _pipeline()._build_run([item])["items"][0]

    def test_precision_and_recall_match(self):
        rag, cmp = self._ragcheck_entry()["metrics"], self._compare_entry()["metrics"]
        assert cmp["b_in_a"] == rag["precision"] == 0.3333
        assert cmp["a_in_b"] == rag["recall"] == 0.6667

    def test_the_verdict_arrays_match(self):
        rag, cmp = self._ragcheck_entry(), self._compare_entry()

        def verdicts(cells):
            return [c["verdict"] for c in cells]

        assert verdicts(cmp["a2b"]) == verdicts(rag["answer2response"])
        assert verdicts(cmp["b2a"]) == verdicts(rag["response2answer"])


# ── Retry marker ─────────────────────────────────────────────────────────────

class TestRetryMarker:

    def _item(self):
        item = _checked_item()
        item[A_KG][1][f"{CHK}_b2a_retry"] = 2      # "flowing north", Neutral
        item[B_KG][1][f"{CHK}_a2b_retry"] = 1      # "7,000 km long", Contradiction
        return item

    def test_cells_carry_the_round_sparsely(self, pipeline):
        entry = pipeline._build_run([self._item()])["items"][0]
        assert entry["b2a"][1]["retry"] == 2
        assert entry["a2b"][1]["retry"] == 1
        assert "retry" not in entry["b2a"][0]

    def test_meta_names_the_rounds(self, pipeline):
        from claimlens.models import DEFAULT_RETRY_ROUNDS
        from claimlens.utils import describe_retry_rounds
        run = pipeline._build_run([self._item()])
        assert run["_meta"]["retry_rounds"] == describe_retry_rounds(DEFAULT_RETRY_ROUNDS)

    def test_findings_carry_the_round(self, pipeline):
        f = pipeline._build_findings(pipeline._build_run([self._item()])["items"])
        assert f["a_not_in_b"][0]["retry"] == 2
        assert f["b_contradicted"][0]["retry"] == 1
        assert "retry" not in f["a_contradicted"][0]
