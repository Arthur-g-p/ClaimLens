"""Unit tests for ExtractorEvaluator — validation, classification, LLM matching, result building."""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from claimlens.eval.extractoreval import (
    ExtractorEvaluator,
    _ItemBucket,
    _ItemMatchResult,
)
from claimlens.exceptions import InvalidInputError
from claimlens.models import ExtractorEvalResult

# A finding names its claims as s/p/o, the triplet a repair takes over.
ABC = {"subject": "a", "predicate": "b", "object": "c"}
DEF = {"subject": "d", "predicate": "e", "object": "f"}
XYZ = {"subject": "x", "predicate": "y", "object": "z"}


# ── Fixtures ─────────────────────────────────────────────────────────────────

GT_KEY = "claude2_response_kg"
PRED_KEY = "test-model_response_kg"

def _make_triplet(s, p, o, human_label=None, aliases=None):
    """Build a GT triplet dict in legacy format."""
    t = {"triplet": [s, p, o]}
    if human_label:
        t["human_label"] = human_label
    if aliases:
        t["aliases"] = aliases
    return t


def _make_canonical_triplet(s, p, o):
    """Build a predicted triplet dict in canonical format."""
    return {"subject": s, "predicate": p, "object": o}


def _make_item(gt_triplets=None, pred_triplets=None, item_id="test", response="r"):
    """Build a test item."""
    item = {"id": item_id, "question": "q", "response": response, "reference": ["ref"]}
    if gt_triplets is not None:
        item[GT_KEY] = gt_triplets
    if pred_triplets is not None:
        item[PRED_KEY] = pred_triplets
    return item


def _evaluator(**kwargs):
    """Build an evaluator with mocked ExtractionService (no API key needed)."""
    defaults = dict(
        extractor_model="test-model",
        gt_key=GT_KEY,
        checker_model="test-checker",
    )
    defaults.update(kwargs)

    with patch("claimlens.eval.extractoreval.ExtractionService"):
        return ExtractorEvaluator(**defaults)


# ── Test _validate ───────────────────────────────────────────────────────────

class TestValidate:
    """_validate drops items missing a response or the GT key; an empty GT
    list is data (nothing to extract — the annotated no-answer)."""

    def test_valid_items_pass(self):
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=[_make_triplet("a", "b", "c")]),
        ]
        valid = ev._validate(data)
        assert len(valid) == 1

    def test_absent_gt_key_is_missing_data(self):
        """Absent = missing (dropped), same rule as ragcheck's gt_answer."""
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=None),
            _make_item(gt_triplets=[_make_triplet("a", "b", "c")]),
        ]
        valid = ev._validate(data)
        assert len(valid) == 1
        assert ev._dropped_missing_gt == 1

    def test_empty_gt_list_is_the_annotated_no_answer(self):
        """Explicit [] = nothing to extract — kept, it is what makes
        unwarranted answers detectable."""
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=[]),
            _make_item(gt_triplets=[_make_triplet("a", "b", "c")]),
        ]
        valid = ev._validate(data)
        assert len(valid) == 2

    def test_empty_gt_NOT_dropped(self):
        """Items with empty GT list also survive validation."""
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=[]),
        ]
        valid = ev._validate(data)
        assert len(valid) == 1

    def test_marked_items_skipped(self):
        """skip_marked leaves marked items out before anything else."""
        ev = _evaluator(skip_marked="cut")
        cut = _make_item(gt_triplets=[_make_triplet("a", "b", "c")], item_id="cut")
        cut["cut"] = ["D-011 absence"]
        data = [cut, _make_item(gt_triplets=[_make_triplet("a", "b", "c")])]
        valid = ev._validate(data)
        assert [i["id"] for i in valid] == ["test"]
        assert ev._skipped_marked == 1

    def test_marked_items_kept_without_skip_marked(self):
        ev = _evaluator()
        cut = _make_item(gt_triplets=[_make_triplet("a", "b", "c")])
        cut["cut"] = ["D-011 absence"]
        assert len(ev._validate([cut])) == 1

    def test_missing_response_dropped(self):
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=[_make_triplet("a", "b", "c")], response=""),
        ]
        with pytest.raises(InvalidInputError):
            ev._validate(data)

    def test_all_missing_response_raises(self):
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=[_make_triplet("a", "b", "c")], response=""),
            _make_item(gt_triplets=None, response=""),
        ]
        with pytest.raises(InvalidInputError, match="No evaluable items"):
            ev._validate(data)


class TestCorpusGtGuard:
    """No item carrying the GT key at all = wrong file or wrong --gt-key.

    Presence of the KEY is what counts. An empty list is a deliberate
    abstention trap and must never trip the guard or the warning.
    """

    def test_no_item_has_gt_key_raises(self):
        ev = _evaluator()
        data = [_make_item(gt_triplets=None), _make_item(gt_triplets=None)]
        with pytest.raises(InvalidInputError, match="No ground truth found"):
            ev._validate(data)

    def test_wrong_gt_key_raises(self):
        """GT present under another key — the real-world --gt-key mistake."""
        ev = _evaluator(gt_key="some_other_key")
        data = [_make_item(gt_triplets=[_make_triplet("a", "b", "c")])]
        with pytest.raises(InvalidInputError, match="some_other_key"):
            ev._validate(data)

    def test_all_empty_gt_lists_do_NOT_raise(self):
        """An all-abstention corpus is a legitimate eval, not a broken input."""
        ev = _evaluator()
        data = [_make_item(gt_triplets=[]), _make_item(gt_triplets=[])]
        valid = ev._validate(data)
        assert len(valid) == 2

    def test_empty_gt_not_counted_as_missing(self):
        """Abstentions carry the key (an empty list), so nothing is dropped."""
        ev = _evaluator()
        data = [_make_item(gt_triplets=[]), _make_item(gt_triplets=[])]
        ev._validate(data)
        assert ev._dropped_missing_gt == 0

    def test_all_gt_keys_absent_is_a_wrong_file(self):
        """Every response-bearing item lacks the key → wrong file / wrong
        --gt-key, caught before any LLM call."""
        ev = _evaluator()
        with pytest.raises(InvalidInputError, match="No ground truth found"):
            ev._validate([_make_item(gt_triplets=None), _make_item(gt_triplets=None)])


class TestDataTreeRendering:
    """_log_data_pre reports each drop reason only when it happened."""

    def test_reports_missing_gt_drops(self):
        ev = _evaluator()
        ev._dropped_missing_response = 0
        ev._dropped_missing_gt = 1
        with patch("claimlens.eval.extractoreval.logger") as mock_log:
            ev._log_data_pre(6, 1, 5)
        lines = [str(c) for c in mock_log.info.call_args_list]
        assert any("missing GT key" in l and "1" in l for l in lines)
        assert not any("missing response" in l for l in lines)

    def test_silent_when_nothing_dropped(self):
        ev = _evaluator()
        ev._dropped_missing_response = 0
        ev._dropped_missing_gt = 0
        with patch("claimlens.eval.extractoreval.logger") as mock_log:
            ev._log_data_pre(5, 0, 5)
        lines = [str(c) for c in mock_log.info.call_args_list]
        assert not any("dropped" in l for l in lines)


# ── Test pred_key / gt_key collision guard ───────────────────────────────────

class TestPredKeyCollisionGuard:
    """The derived predicted-key (f'{extractor_model}_response_kg') must never
    equal the GT key — otherwise extraction targets the GT slot and the
    evaluator matches ground truth against itself, reporting a false perfect score."""

    def test_collision_via_extractor_model_raises(self):
        """extractor_model='claude2' derives 'claude2_response_kg', colliding with the default gt_key."""
        with pytest.raises(InvalidInputError, match="collides"):
            _evaluator(extractor_model="claude2", gt_key="claude2_response_kg")

    def test_collision_via_gt_key_raises(self):
        """A gt_key set to match the derived predicted-key also collides."""
        with pytest.raises(InvalidInputError, match="collides"):
            _evaluator(extractor_model="test-model", gt_key="test-model_response_kg")

    def test_no_collision_constructs_ok(self):
        """Distinct extractor_model and gt_key construct without raising."""
        ev = _evaluator(extractor_model="gemini-3.1", gt_key="claude2_response_kg")
        assert ev._pred_key == "gemini-3.1_response_kg"
        assert ev._pred_key != ev._gt_key


# ── Test _classify (post-extraction) ─────────────────────────────────────────

class TestClassify:
    """_classify should sort items into four buckets after extraction."""

    def test_valid_item(self):
        """Item with both GT and predictions → to_compare."""
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=[_make_triplet("a", "b", "c")],
            pred_triplets=[_make_canonical_triplet("a", "b", "c")],
        )]
        buckets = ev._classify(data)
        assert len(buckets.to_compare) == 1
        assert len(buckets.abstention_misread) == 0
        assert len(buckets.answer_missed) == 0
        assert len(buckets.abstention_recognized) == 0

    def test_abstention_misread(self):
        """Item with predictions but no GT → unwarranted answer."""
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=None,
            pred_triplets=[_make_canonical_triplet("a", "b", "c")],
        )]
        buckets = ev._classify(data)
        assert len(buckets.abstention_misread) == 1

    def test_answer_missed(self):
        """Item with GT but no predictions → unwarranted abstention."""
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=[_make_triplet("a", "b", "c")],
            pred_triplets=None,
        )]
        buckets = ev._classify(data)
        assert len(buckets.answer_missed) == 1

    def test_skipped(self):
        """Item with neither GT nor predictions → abstention_recognized."""
        ev = _evaluator()
        data = [_make_item(gt_triplets=None, pred_triplets=None)]
        buckets = ev._classify(data)
        assert len(buckets.abstention_recognized) == 1

    def test_empty_gt_treated_as_no_gt(self):
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=[],
            pred_triplets=[_make_canonical_triplet("a", "b", "c")],
        )]
        buckets = ev._classify(data)
        assert len(buckets.abstention_misread) == 1

    def test_empty_pred_treated_as_no_pred(self):
        ev = _evaluator()
        data = [_make_item(
            gt_triplets=[_make_triplet("a", "b", "c")],
            pred_triplets=[],
        )]
        buckets = ev._classify(data)
        assert len(buckets.answer_missed) == 1

    def test_mixed_classification(self):
        ev = _evaluator()
        data = [
            _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                       pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            _make_item(gt_triplets=None, pred_triplets=[_make_canonical_triplet("x", "y", "z")]),
            _make_item(gt_triplets=[_make_triplet("d", "e", "f")], pred_triplets=None),
            _make_item(gt_triplets=None, pred_triplets=None),
        ]
        buckets = ev._classify(data)
        assert len(buckets.to_compare) == 1
        assert len(buckets.abstention_misread) == 1
        assert len(buckets.answer_missed) == 1
        assert len(buckets.abstention_recognized) == 1


# ── Test _triplet_to_str ─────────────────────────────────────────────────────

class TestTripletToStr:
    def test_canonical_format(self):
        t = {"subject": "dogs", "predicate": "are", "object": "animals"}
        assert ExtractorEvaluator._triplet_to_str(t) == "dogs are animals"

    def test_legacy_dict_raises(self):
        # Triplets are canonicalized at ingestion; an un-canonicalized legacy
        # dict reaching this method is a contract violation, not a fallback.
        with pytest.raises(KeyError):
            ExtractorEvaluator._triplet_to_str({"triplet": ["dogs", "are", "animals"]})


# ── Test _measure_duplicates (orthogonal, read-only) ──────────────────────────

class TestMeasureDuplicates:
    def test_none_when_no_predictions(self):
        ev = _evaluator()
        valid = [_make_item(gt_triplets=[_make_canonical_triplet("a", "b", "c")])]
        assert ev._measure_duplicates(valid) is None

    def test_counts_and_lists_duplicates(self):
        ev = _evaluator()
        valid = [_make_item(item_id="x", pred_triplets=[
            _make_canonical_triplet("a", "b", "c"),
            _make_canonical_triplet("a", "b", "c"),
            _make_canonical_triplet("d", "e", "f"),
        ])]
        d = ev._measure_duplicates(valid)
        assert d["predicted_claims"] == 3
        assert d["duplicate_claims"] == 1
        assert d["unique_claims"] == 2
        assert d["duplicate_rate"] == round(1 / 3, 4)
        assert d["items"][0]["id"] == "x"
        assert d["items"][0]["duplicates"] == ["a b c"]

    def test_no_duplicates_gives_empty_listing(self):
        ev = _evaluator()
        valid = [_make_item(pred_triplets=[
            _make_canonical_triplet("a", "b", "c"),
            _make_canonical_triplet("d", "e", "f"),
        ])]
        d = ev._measure_duplicates(valid)
        assert d["duplicate_claims"] == 0
        assert d["items"] == []

    def test_read_only_never_mutates_predictions(self):
        ev = _evaluator()
        preds = [_make_canonical_triplet("a", "b", "c"), _make_canonical_triplet("a", "b", "c")]
        valid = [_make_item(pred_triplets=preds)]
        ev._measure_duplicates(valid)
        assert len(preds) == 2  # the eval never deduplicates its data


# ── Test _count_pass (pure verdict counting, one pass) ───────────────────────

V_KEY = "test-checker_checker_verdict"
E_KEY = "test-checker_checker_explanation"


def _checked(*verdicts, explanation=None):
    """Build checked triplets carrying the given verdicts (None = no verdict key)."""
    out = []
    for v in verdicts:
        t = {}
        if v is not None:
            t[V_KEY] = v
        if explanation is not None:
            t[E_KEY] = explanation
        out.append(t)
    return out


def _originals(n):
    return [_make_canonical_triplet(f"s{i}", f"p{i}", f"o{i}") for i in range(n)]


class TestCountPass:
    """Three-way verdict split: Entailment / judged miss / unjudged (None)."""

    def _run(self, checked, originals, field="gt_triplet"):
        return ExtractorEvaluator._count_pass(checked, originals, V_KEY, E_KEY, field)

    def test_all_entailment(self):
        entailed, misses, unjudged = self._run(
            _checked("Entailment", "Entailment"), _originals(2))
        assert entailed == 2
        assert misses == []
        assert unjudged == []

    def test_real_verdicts_are_misses(self):
        entailed, misses, unjudged = self._run(
            _checked("Entailment", "Neutral", "Contradiction"), _originals(3))
        assert entailed == 1
        assert [m["verdict"] for m in misses] == ["Neutral", "Contradiction"]
        assert unjudged == []

    def test_miss_carries_triplet_string_and_field_name(self):
        _, misses, _ = self._run(_checked("Neutral"), _originals(1), field="pred_triplet")
        assert misses[0]["pred_triplet"] == "s0 p0 o0"

    def test_explanation_used_as_reason(self):
        _, misses, _ = self._run(
            _checked("Neutral", explanation="not in reference"), _originals(1))
        assert misses[0]["reason"] == "not in reference"

    def test_verdict_falls_back_as_reason(self):
        _, misses, _ = self._run(_checked("Neutral"), _originals(1))
        assert misses[0]["reason"] == "Neutral"

    def test_none_verdict_is_unjudged_never_a_miss(self):
        """The fix: a None verdict (checker failure — nobody judged the claim)
        is no evidence about the extractor. It leaves numerator AND
        denominator instead of being charged as a miss."""
        entailed, misses, unjudged = self._run(
            _checked("Entailment", None), _originals(2))
        assert entailed == 1
        assert misses == []
        assert len(unjudged) == 1
        assert unjudged[0]["gt_triplet"] == "s1 p1 o1"
        assert unjudged[0]["cause"] == "checker_failure"

    def test_missing_verdict_key_is_unjudged(self):
        """A triplet the checker never touched (no verdict key at all) is
        indistinguishable from an explicit None — both are unjudged."""
        entailed, misses, unjudged = self._run(_checked(None), _originals(1))
        assert (entailed, misses) == (0, [])
        assert len(unjudged) == 1

    def test_all_unjudged_zero_denominator(self):
        """Checker failed everything: nothing entailed, nothing missed —
        the pass contributes an empty (0/0) measurement, not a zero score."""
        entailed, misses, unjudged = self._run(_checked(None, None), _originals(2))
        assert (entailed, len(misses), len(unjudged)) == (0, 0, 2)

    def test_three_way_partition_is_exhaustive(self):
        """Every claim lands in exactly one bucket — the counts must always
        re-add to the number of issued claims."""
        checked = _checked("Entailment", "Neutral", None, "Contradiction", None)
        entailed, misses, unjudged = self._run(checked, _originals(5))
        assert entailed + len(misses) + len(unjudged) == 5


# ── Test _build_result ───────────────────────────────────────────────────────

class TestBuildResult:
    """Test the encapsulated _build_result method."""

    def test_basic_metrics(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                           pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            ],
            abstention_misread=[],
            answer_missed=[],
            abstention_recognized=[],
        )
        item_results = [_ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0, false_positives=[], false_negatives=[])]

        result = ev._build_result(item_results, buckets, total_items=1)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.recall_counts["covered"] == 1
        assert result.recall_counts["denominator"] == 1
        assert result.precision_counts["supported"] == 1
        assert result.precision_counts["denominator"] == 1
        assert result.checker_failures["count"] == 0
        assert result.to_compare_items == 1
        assert result.abstention_handling["abstentions_recognized"] == 0

    def test_answer_missed_penalty(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                           pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            ],
            abstention_misread=[],
            answer_missed=[
                _make_item(gt_triplets=[_make_triplet("d", "e", "f"),
                                        _make_triplet("g", "h", "i")]),
            ],
            abstention_recognized=[],
        )
        item_results = [_ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0, false_positives=[], false_negatives=[])]

        result = ev._build_result(item_results, buckets, total_items=2)
        rc = result.recall_counts
        assert rc["covered"] == 1
        assert rc["missed"] == 0
        assert rc["answer_missed_penalty"] == 2  # penalty IS in the denominator
        assert rc["denominator"] == 3
        assert result.recall == round(1 / 3, 4)
        assert result.abstention_handling["answers_missed"] == 1

    def test_abstention_misread_fp_penalty(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                           pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            ],
            abstention_misread=[
                _make_item(gt_triplets=None,
                           pred_triplets=[_make_canonical_triplet("x", "y", "z")]),
            ],
            answer_missed=[],
            abstention_recognized=[],
        )
        item_results = [_ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0, false_positives=[], false_negatives=[])]

        result = ev._build_result(item_results, buckets, total_items=2)
        pc = result.precision_counts
        assert pc["supported"] == 1
        assert pc["unsupported"] == 0
        assert pc["abstention_misread_penalty"] == 1  # penalty IS in the denominator
        assert pc["denominator"] == 2
        assert result.precision == 0.5
        assert result.abstention_handling["abstentions_misread"] == 1

    def test_zero_items(self):
        """Empty denominator → None, never 0.0: nothing judged is not a score."""
        ev = _evaluator()
        buckets = _ItemBucket(to_compare=[], abstention_misread=[], answer_missed=[], abstention_recognized=[])
        result = ev._build_result([], buckets, total_items=0)
        assert result.precision is None
        assert result.recall is None
        assert result.f1 is None

    def test_all_unjudged_yields_null_not_zero(self):
        """Checker failed everything: 0/0 on both sides must be None — a 0.0
        would give the worst possible score for zero information."""
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                           pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            ],
            abstention_misread=[],
            answer_missed=[],
            abstention_recognized=[],
        )
        item_results = [_ItemMatchResult(
            tp_recall=0, tp_precision=0, fp=0, fn=0,
            false_positives=[], false_negatives=[],
            unjudged_gt=[{"gt_triplet": "a b c", "cause": "checker_failure"}],
            unjudged_pred=[{"pred_triplet": "a b c", "cause": "checker_failure"}],
        )]
        result = ev._build_result(item_results, buckets, total_items=1)
        assert result.precision is None
        assert result.recall is None
        assert result.f1 is None
        assert result.checker_failures == {
            "count": 2, "issued_verdicts": 2, "rate": 1.0,
            "items_affected": 1, "unjudged_gt": 1, "unjudged_pred": 1,
        }

    def test_unjudged_excluded_from_denominator(self):
        """1 covered + 1 unjudged GT claim → recall 1/1, not 1/2; the unjudged
        claim shows up in checker_failures instead."""
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet("a", "b", "c"),
                                        _make_triplet("d", "e", "f")],
                           pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            ],
            abstention_misread=[],
            answer_missed=[],
            abstention_recognized=[],
        )
        item_results = [_ItemMatchResult(
            tp_recall=1, tp_precision=1, fp=0, fn=0,
            false_positives=[], false_negatives=[],
            unjudged_gt=[{"gt_triplet": "d e f", "cause": "checker_failure"}],
        )]
        result = ev._build_result(item_results, buckets, total_items=1)
        assert result.recall == 1.0
        assert result.recall_counts["denominator"] == 1
        assert result.recall_counts["unjudged"] == 1
        assert result.recall_counts["total_gt_claims"] == 2
        assert result.checker_failures["count"] == 1

    def test_counts_are_exhaustive_partitions(self):
        """total = judged buckets + penalty + unjudged, denominator = total - unjudged."""
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet(c, c, c) for c in "abcd"],
                           pred_triplets=[_make_canonical_triplet(c, c, c) for c in "abc"]),
            ],
            abstention_misread=[
                _make_item(gt_triplets=None,
                           pred_triplets=[_make_canonical_triplet("x", "y", "z")]),
            ],
            answer_missed=[
                _make_item(gt_triplets=[_make_triplet("d", "e", "f")]),
            ],
            abstention_recognized=[],
        )
        item_results = [_ItemMatchResult(
            tp_recall=2, tp_precision=1, fp=1, fn=1,
            false_positives=[{"pred_triplet": "b b b", "verdict": "Neutral", "reason": "r"}],
            false_negatives=[{"gt_triplet": "c c c", "verdict": "Contradiction", "reason": "r"}],
            unjudged_gt=[{"gt_triplet": "d d d", "cause": "checker_failure"}],
            unjudged_pred=[{"pred_triplet": "c c c", "cause": "checker_failure"}],
        )]
        result = ev._build_result(item_results, buckets, total_items=3)
        rc, pc = result.recall_counts, result.precision_counts
        assert (rc["covered"] + rc["missed"] + rc["answer_missed_penalty"]
                + rc["unjudged"]) == rc["total_gt_claims"] == 5
        assert rc["denominator"] == rc["total_gt_claims"] - rc["unjudged"] == 4
        assert (pc["supported"] + pc["unsupported"] + pc["abstention_misread_penalty"]
                + pc["unjudged"]) == pc["total_pred_claims"] == 4
        assert pc["denominator"] == pc["total_pred_claims"] - pc["unjudged"] == 3
        # Totals must equal what the extraction stats report
        assert rc["total_gt_claims"] == result.gt_stats["total_triplets"]
        assert pc["total_pred_claims"] == result.pred_stats["total_triplets"]
        # Issued verdicts: judged + unjudged, penalties never sent to the checker
        assert result.checker_failures["issued_verdicts"] == 7


class TestRetryInFindings:

    def test_unsupported_names_its_round(self):
        ev = _evaluator()
        item = _make_item(gt_triplets=[_make_canonical_triplet("a", "b", "c")],
                          pred_triplets=[_make_canonical_triplet("x", "y", "z")],
                          item_id="r1")
        buckets = _ItemBucket(
            to_compare=[item], abstention_misread=[],
            answer_missed=[], abstention_recognized=[],
        )
        item_results = [_ItemMatchResult(
            tp_recall=0, tp_precision=0, fp=1, fn=1,
            false_positives=[], false_negatives=[],
            pred2gt=[{"verdict": "Neutral", "explanation": "e"}],
            gt2pred=[{"verdict": "Neutral", "explanation": "e", "retry": 1}],
        )]
        findings = ev._build_findings(ev._build_items(buckets, item_results))
        assert findings["unsupported"][0]["retry"] == 1
        assert "retry" not in findings["missed"][0]


class TestUnjudgedInFindings:
    """Checker failures are not disagreements, but affected items must stay
    identifiable in the findings with their unjudged claims."""

    def test_item_with_only_unjudged_appears(self):
        ev = _evaluator()
        item = _make_item(gt_triplets=[_make_canonical_triplet("a", "b", "c")],
                          pred_triplets=[_make_canonical_triplet("a", "b", "c")],
                          item_id="u1")
        buckets = _ItemBucket(
            to_compare=[item], abstention_misread=[],
            answer_missed=[], abstention_recognized=[],
        )
        item_results = [_ItemMatchResult(
            tp_recall=0, tp_precision=1, fp=0, fn=0,
            false_positives=[], false_negatives=[],
            unjudged_gt=[{"gt_triplet": "a b c", "cause": "checker_failure"}],
            pred2gt=[{"verdict": None, "explanation": None}],
            gt2pred=[{"verdict": "Entailment", "explanation": "e"}],
        )]
        items = ev._build_items(buckets, item_results)
        findings = ev._build_findings(items)
        assert findings["unjudged"] == [
            {"id": "u1", "question": "q", "claim": ABC, "side": "gt", "cause": "checker_failure",
             "response": "r", "gt_claims": [ABC], "pred_claims": [ABC]}]
        assert findings["missed"] == [] and findings["unsupported"] == []

    def test_perfect_match_is_in_items_but_not_in_findings(self):
        ev = _evaluator()
        item = _make_item(gt_triplets=[_make_canonical_triplet("a", "b", "c")],
                          pred_triplets=[_make_canonical_triplet("a", "b", "c")])
        buckets = _ItemBucket(
            to_compare=[item], abstention_misread=[],
            answer_missed=[], abstention_recognized=[],
        )
        item_results = [_ItemMatchResult(
            tp_recall=1, tp_precision=1, fp=0, fn=0,
            false_positives=[], false_negatives=[],
            pred2gt=[{"verdict": "Entailment", "explanation": "e"}],
            gt2pred=[{"verdict": "Entailment", "explanation": "e"}],
        )]
        items = ev._build_items(buckets, item_results)
        assert [i["bucket"] for i in items] == ["compared"]
        # ragcheck's shape: clean s/p/o claims, verdict cells in parallel arrays
        assert items[0]["gt_claims"] == items[0]["pred_claims"] == [
            {"subject": "a", "predicate": "b", "object": "c"}]
        assert items[0]["pred2gt"] == items[0]["gt2pred"] == [
            {"verdict": "Entailment", "explanation": "e"}]
        assert all(v == [] for v in ev._build_findings(items).values())

    async def test_matching_keeps_the_checker_error_cause(self):
        """The service persists why a verdict is null; the record must not
        drop it (docs/outcome_markers.md: a null verdict is never opaque)."""
        class FakeService:
            def __init__(self, **kwargs):
                pass

            async def run(self, items):
                for it in items:
                    for t in it["_exteval_response_kg"]:
                        if t["subject"] == "bad":
                            t["test-checker_checker_verdict"] = None
                            t["test-checker_checker_explanation"] = None
                            t["test-checker_checker_error"] = "context_too_long"
                        elif t["subject"] == "late":
                            t["test-checker_checker_verdict"] = "Neutral"
                            t["test-checker_checker_explanation"] = "e"
                            t["test-checker_checker_retry"] = 2
                        else:
                            t["test-checker_checker_verdict"] = "Entailment"
                            t["test-checker_checker_explanation"] = "e"
                return items

        ev = _evaluator(match_mode="many-to-many")
        item = _make_item(gt_triplets=[_make_canonical_triplet("a", "b", "c"),
                                       _make_canonical_triplet("bad", "x", "y")],
                          pred_triplets=[_make_canonical_triplet("a", "b", "c"),
                                         _make_canonical_triplet("late", "x", "y")])
        with patch("claimlens.eval.extractoreval.CheckingService", FakeService):
            [result] = await ev._match_all_llm([item])
        assert result.pred2gt == [
            {"verdict": "Entailment", "explanation": "e"},
            {"verdict": None, "explanation": None, "error": "context_too_long"},
        ]
        assert "error" not in result.gt2pred[0]
        assert result.gt2pred[1]["retry"] == 2
        assert "retry" not in result.gt2pred[0]

    def test_unjudged_finding_names_the_real_cause(self):
        ev = _evaluator()
        item = _make_item(gt_triplets=[_make_canonical_triplet("a", "b", "c")],
                          pred_triplets=[_make_canonical_triplet("a", "b", "c")],
                          item_id="u2")
        buckets = _ItemBucket(
            to_compare=[item], abstention_misread=[],
            answer_missed=[], abstention_recognized=[],
        )
        item_results = [_ItemMatchResult(
            tp_recall=0, tp_precision=1, fp=0, fn=0,
            false_positives=[], false_negatives=[],
            unjudged_gt=[{"gt_triplet": "a b c", "cause": "checker_failure"}],
            pred2gt=[{"verdict": None, "explanation": None, "error": "context_too_long"}],
            gt2pred=[{"verdict": "Entailment", "explanation": "e"}],
        )]
        findings = ev._build_findings(ev._build_items(buckets, item_results))
        assert findings["unjudged"] == [
            {"id": "u2", "question": "q", "claim": ABC, "side": "gt",
             "cause": "context_too_long", "response": "r",
             "gt_claims": [ABC], "pred_claims": [ABC]}]



# ── Test extraction-error handling (tooling failures, never abstentions) ─────

ERROR_KEY = "test-model_extraction_error"


class TestExtractionErrors:
    """[] caused by a tooling failure must be excluded from ALL metrics and
    reported as an error rate — not scored as (unwarranted/correct) abstention."""

    def test_errored_items_get_own_bucket(self):
        ev = _evaluator()
        errored = _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                             pred_triplets=[])
        errored[ERROR_KEY] = "parse_failure"
        normal = _make_item(gt_triplets=[_make_triplet("d", "e", "f")],
                            pred_triplets=[_make_canonical_triplet("d", "e", "f")])

        buckets = ev._classify([errored, normal])
        assert buckets.extraction_error == [errored]
        # Crucially NOT scored as unwarranted abstention despite GT + empty pred
        assert buckets.answer_missed == []
        assert buckets.to_compare == [normal]

    def test_error_rate_computed_and_no_fn_penalty(self):
        ev = _evaluator()
        errored = _make_item(gt_triplets=[_make_triplet("a", "b", "c"),
                                          _make_triplet("d", "e", "f")],
                             pred_triplets=[])
        errored[ERROR_KEY] = "parse_failure"
        buckets = _ItemBucket(
            to_compare=[],
            abstention_misread=[],
            answer_missed=[],
            abstention_recognized=[_make_item()],
            extraction_error=[errored],
        )
        result = ev._build_result([], buckets, total_items=2)
        # The errored item's 2 GT triplets added NO FN penalty
        assert result.recall_counts["answer_missed_penalty"] == 0
        assert result.recall_counts["total_gt_claims"] == 0
        assert result.extraction_errors == {
            "count": 1,
            "rate": 0.5,
            "by_cause": {"parse_failure": 1},
        }

    def test_errored_items_listed_in_findings(self):
        ev = _evaluator()
        errored = _make_item(item_id="broken-1")
        errored[ERROR_KEY] = "context_too_long"
        buckets = _ItemBucket(
            to_compare=[], abstention_misread=[], answer_missed=[],
            abstention_recognized=[], extraction_error=[errored],
        )
        items = ev._build_items(buckets, [])
        assert items[0]["bucket"] == "extraction_error"
        assert items[0]["cause"] == "context_too_long"
        findings = ev._build_findings(items)
        assert findings["extraction_failed"] == [
            {"id": "broken-1", "question": "q", "cause": "context_too_long"}]


# ── Test _build_items / _build_findings ────────────────────────────────────────────────

class TestItemsAndFindings:
    """items = the complete record (every bucket); findings = the review
    queue derived from it, each entry tagged with a kind."""

    def test_perfect_match_excluded_from_findings(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[_make_item(
                gt_triplets=[_make_triplet("a", "b", "c")],
                pred_triplets=[_make_canonical_triplet("a", "b", "c")],
            )],
            abstention_misread=[], answer_missed=[], abstention_recognized=[],
        )
        results = [_ItemMatchResult(
            tp_recall=1, tp_precision=1, fp=0, fn=0, false_positives=[], false_negatives=[],
            pred2gt=[{"verdict": "Entailment", "explanation": "e"}],
            gt2pred=[{"verdict": "Entailment", "explanation": "e"}])]
        items = ev._build_items(buckets, results)
        assert len(items) == 1
        assert all(v == [] for v in ev._build_findings(items).values())

    def test_fp_and_fn_become_findings(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[_make_item(
                gt_triplets=[_make_canonical_triplet("a", "b", "c")],
                pred_triplets=[_make_canonical_triplet("x", "y", "z")],
            )],
            abstention_misread=[], answer_missed=[], abstention_recognized=[],
        )
        results = [_ItemMatchResult(
            tp_recall=0, tp_precision=0, fp=1, fn=1,
            false_positives=[{"pred_triplet": "x y z", "verdict": "Neutral", "reason": "r"}],
            false_negatives=[{"gt_triplet": "a b c", "verdict": "Neutral", "reason": "r"}],
            pred2gt=[{"verdict": "Neutral", "explanation": "no match"}],
            gt2pred=[{"verdict": "Neutral", "explanation": "no match"}])]
        items = ev._build_items(buckets, results)
        findings = ev._build_findings(items)
        assert list(findings) == ["missed", "unsupported", "answer_missed",
                                  "abstention_misread", "unjudged", "extraction_failed"]
        # each carries what it is judged against: the response and both claim lists
        both = {"response": "r", "gt_claims": [ABC], "pred_claims": [XYZ]}
        assert findings["missed"] == [
            {"id": "test", "question": "q", "claim": ABC, "verdict": "Neutral", "explanation": "no match",
             **both}]
        assert findings["unsupported"] == [
            {"id": "test", "question": "q", "claim": XYZ, "verdict": "Neutral", "explanation": "no match",
             **both}]

    def test_abstention_misread_findings(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[],
            abstention_misread=[_make_item(
                gt_triplets=None,
                pred_triplets=[_make_canonical_triplet("a", "b", "c")],
            )],
            answer_missed=[], abstention_recognized=[],
        )
        items = ev._build_items(buckets, [])
        assert items[0]["bucket"] == "abstention_misread"
        assert items[0]["pred_claims"] == [{"subject": "a", "predicate": "b", "object": "c"}]
        findings = ev._build_findings(items)
        assert findings["abstention_misread"] == [
            {"id": "test", "question": "q", "response": "r", "claims": [ABC]}]

    def test_answer_missed_findings(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[], abstention_misread=[],
            answer_missed=[_make_item(
                gt_triplets=[_make_canonical_triplet("a", "b", "c"), _make_canonical_triplet("d", "e", "f")],
                pred_triplets=None,
            )],
            abstention_recognized=[],
        )
        items = ev._build_items(buckets, [])
        findings = ev._build_findings(items)
        assert findings["answer_missed"] == [
            {"id": "test", "question": "q", "response": "r", "claims": [ABC, DEF]}]

    def test_recognized_abstention_is_in_items_only(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[], abstention_misread=[], answer_missed=[],
            abstention_recognized=[_make_item(gt_triplets=[], pred_triplets=[], item_id="ok")],
        )
        items = ev._build_items(buckets, [])
        assert items == [{"id": "ok", "question": "q", "response": "r",
                          "bucket": "abstention_recognized", "gt_claims": [], "pred_claims": []}]
        assert all(v == [] for v in ev._build_findings(items).values())



# ── Integration-lite tests ───────────────────────────────────────────────────

class TestEvaluateIntegration:
    """Test evaluate() flow with mocked extraction service."""

    def test_perfect_score_with_mocked_extraction(self):
        """Extraction produces exact matches → P=R=F1=1.0."""
        ev = _evaluator()
        
        async def mock_match_all_llm(items):
            return [_ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0, false_positives=[], false_negatives=[])]
        ev._match_all_llm = AsyncMock(side_effect=mock_match_all_llm)

        data = [
            _make_item(
                gt_triplets=[_make_triplet("a", "b", "c")],
                item_id="1",
            ),
        ]

        # Mock extraction: service.run() adds pred triplets to items
        async def mock_run(items):
            for item in items:
                item[PRED_KEY] = [_make_canonical_triplet("a", "b", "c")]

        ev._extraction_service.run = AsyncMock(side_effect=mock_run)

        with patch.object(ev, "_log_data_pre"), \
             patch.object(ev, "_log_eval_config"), \
             patch.object(ev, "_log_eval_results"), \
             patch.object(ev, "_log_done"):
            record, findings = ev.run_sync(data)

        assert list(record) == ["_meta", "metrics", "variance", "runs"]
        assert record["metrics"]["precision"] == 1.0
        assert record["metrics"]["recall"] == 1.0
        assert record["metrics"]["f1"] == 1.0
        assert record["_meta"]["report_type"] == "extractor_eval"
        from claimlens.models import DEFAULT_RETRY_ROUNDS
        from claimlens.utils import describe_retry_rounds
        assert record["_meta"]["retry_rounds"] == describe_retry_rounds(DEFAULT_RETRY_ROUNDS)
        assert record["_meta"]["runs"] == 1 and len(record["runs"]) == 1
        run = record["runs"][0]
        assert list(run) == ["_meta", "metrics", "counts", "items"]
        assert run["counts"]["recall"]["covered"] == 1
        assert all(v == [] for v in findings["runs"][0]["findings"].values())

    def test_answer_missed_from_extraction(self):
        """Extraction produces empty results → unjustified abstention."""
        ev = _evaluator()

        data = [
            _make_item(
                gt_triplets=[_make_triplet("a", "b", "c"), _make_triplet("d", "e", "f")],
                item_id="1",
            ),
        ]

        # Mock extraction: service.run() produces nothing (abstention)
        async def mock_run(items):
            for item in items:
                item[PRED_KEY] = []

        ev._extraction_service.run = AsyncMock(side_effect=mock_run)

        with patch.object(ev, "_log_data_pre"), \
             patch.object(ev, "_log_eval_config"), \
             patch.object(ev, "_log_eval_results"), \
             patch.object(ev, "_log_done"):
            record, findings = ev.run_sync(data)

        run = record["runs"][0]
        assert record["metrics"]["recall"] == 0.0  # measured: 0 of 2 GT claims covered
        assert run["counts"]["recall"]["covered"] == 0
        assert run["counts"]["recall"]["answer_missed_penalty"] == 2
        assert run["counts"]["recall"]["denominator"] == 2
        assert record["metrics"]["precision"] is None  # no predictions → nothing to judge
        assert run["counts"]["abstention_handling"]["answers_missed"] == 1
        queue = findings["runs"][0]["findings"]
        assert queue["answer_missed"] == [
            {"id": "1", "question": "q", "response": "r", "claims": [ABC, DEF]}]

    def test_abstention_misread_from_extraction(self):
        """GT is an explicit empty list (nothing to extract) but extraction
        produces triplets → unwarranted answer."""
        ev = _evaluator()

        data = [
            _make_item(
                gt_triplets=[],
                item_id="1",
            ),
            # Second no-answer item with no predictions: a justified
            # abstention, so the counts below are unchanged.
            _make_item(
                gt_triplets=[],
                item_id="2",
            ),
        ]

        # Mock extraction: service.run() produces triplets on the no-GT item
        async def mock_run(items):
            for item in items:
                if item["id"] == "1":
                    item[PRED_KEY] = [_make_canonical_triplet("x", "y", "z")]

        ev._extraction_service.run = AsyncMock(side_effect=mock_run)

        with patch.object(ev, "_log_data_pre"), \
             patch.object(ev, "_log_eval_config"), \
             patch.object(ev, "_log_eval_results"), \
             patch.object(ev, "_log_done"):
            record, findings = ev.run_sync(data)

        run = record["runs"][0]
        assert run["counts"]["abstention_handling"]["abstentions_misread"] == 1
        assert run["counts"]["precision"]["abstention_misread_penalty"] == 1
        assert record["metrics"]["precision"] == 0.0  # measured: 1 penalty claim, 0 supported
        # both no-GT items are in the record; only the misread one is a finding
        assert sorted(i["bucket"] for i in run["items"]) == ["abstention_misread", "abstention_recognized"]
        queue = findings["runs"][0]["findings"]
        assert queue["abstention_misread"] == [
            {"id": "1", "question": "q", "response": "r", "claims": [XYZ]}]


# ── Smoke: the full EXTRACTOR EVAL print path ────────────────────────────────

class TestLogEvalResultsSmoke:
    """The end-to-end tests patch _log_eval_results away, so a NameError in
    the print path never surfaced. Render every block once, no mocks."""

    def test_full_block_renders(self):
        ev = _evaluator()
        buckets = _ItemBucket(
            to_compare=[
                _make_item(gt_triplets=[_make_triplet("a", "b", "c")],
                           pred_triplets=[_make_canonical_triplet("a", "b", "c")]),
            ],
            abstention_misread=[_make_item(
                gt_triplets=[], pred_triplets=[_make_canonical_triplet("x", "y", "z")])],
            answer_missed=[_make_item(gt_triplets=[_make_canonical_triplet("d", "e", "f")],
                                      pred_triplets=[])],
            abstention_recognized=[_make_item(gt_triplets=[], pred_triplets=[])],
        )
        item_results = [_ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0,
                                         false_positives=[], false_negatives=[])]
        result = ev._build_result(item_results, buckets, total_items=4)
        ev._log_eval_results(result)   # must not raise
        import time
        ev._started_at, ev._started_perf = "2026-01-01T00:00:00", time.perf_counter()
        run_doc, run_findings = ev._run_documents(result, buckets, item_results, 4)
        ev._log_done(run_doc)
        assert {i["bucket"] for i in run_doc["items"]} == {
            "compared", "abstention_misread", "answer_missed", "abstention_recognized"}
        assert run_doc["counts"]["reliability"]["atomization"] == {
            "measured": False, "reason": "no --atomizer-model"}


class TestQuestionIf:
    """--question-if in the eval: a marked item without a question is dropped
    and counted; an item extracted with the question carries question_given."""

    def _run(self, ev, data):
        ev._match_all_llm = AsyncMock(side_effect=lambda items: [
            _ItemMatchResult(tp_recall=1, tp_precision=1, fp=0, fn=0,
                             false_positives=[], false_negatives=[]) for _ in items])

        async def mock_run(items):
            for item in items:
                item[PRED_KEY] = [_make_canonical_triplet("a", "b", "c")]

        ev._extraction_service.run = AsyncMock(side_effect=mock_run)
        with patch.object(ev, "_log_data_pre"), patch.object(ev, "_log_eval_config"), \
             patch.object(ev, "_log_eval_results"), patch.object(ev, "_log_done"):
            return ev.run_sync(data)

    def _item(self, item_id, **extra):
        item = _make_item(gt_triplets=[_make_triplet("a", "b", "c")], item_id=item_id)
        item.update(extra)
        return item

    def test_off_by_default_and_passed_through(self):
        with patch("claimlens.eval.extractoreval.ExtractionService") as cls:
            ExtractorEvaluator(extractor_model="test-model", gt_key=GT_KEY, checker_model="c")
        assert cls.call_args.kwargs["question_if"] is None
        with patch("claimlens.eval.extractoreval.ExtractionService") as cls:
            ExtractorEvaluator(extractor_model="test-model", gt_key=GT_KEY, checker_model="c",
                               question_if="needs_question")
        assert cls.call_args.kwargs["question_if"] == "needs_question"

    def test_marked_without_question_dropped(self):
        ev = _evaluator(question_if="needs_question")
        no_q = self._item("x", needs_question=True)
        del no_q["question"]
        valid = ev._validate([no_q, self._item("y", needs_question=True), self._item("z")])
        assert [i["id"] for i in valid] == ["y", "z"]
        assert ev._dropped_missing_question == 1

    def test_without_the_option_nothing_is_dropped(self):
        ev = _evaluator()
        no_q = self._item("x", needs_question=True)
        del no_q["question"]
        assert len(ev._validate([no_q])) == 1

    def test_record_counts_and_question_given(self):
        ev = _evaluator(question_if="needs_question")
        no_q = self._item("x", needs_question=True)
        del no_q["question"]
        record, _ = self._run(ev, [no_q, self._item("y", needs_question=True),
                                   self._item("z", needs_question=False)])
        run = record["runs"][0]
        assert run["counts"]["data"]["dropped_no_question"] == 1
        by_id = {i["id"]: i for i in run["items"]}
        assert by_id["y"]["question_given"] is True
        assert "question_given" not in by_id["z"]
        assert set(by_id) == {"y", "z"}

    def test_data_block_line(self, caplog):
        import logging
        ev = _evaluator(question_if="needs_question")
        ev._dropped_missing_question = 2
        with caplog.at_level(logging.INFO, logger="claimlens"):
            ev._log_data_pre(5, 2, 3)
        assert "├─ dropped:  2  (marked 'needs_question', no 'question')" in caplog.text

    def test_config_line(self, caplog):
        import logging
        ev = _evaluator(question_if="needs_question")
        ev._extraction_service.base_url = None
        with caplog.at_level(logging.INFO, logger="claimlens"):
            ev._log_eval_config([{"needs_question": True}, {"needs_question": False}, {}])
        assert "Question:    1 of 3 items get the question prompt ('needs_question')" in caplog.text


class TestQuestionIfCli:

    def _invoke(self, tmp_path, monkeypatch, command, target, *extra):
        import json
        from typer.testing import CliRunner
        from claimlens.cli import app
        seen = {}

        class _Built:
            def __init__(self, **kwargs):
                seen.update(kwargs)

            def run_sync(self, data):
                return data

        monkeypatch.setattr(target, _Built)
        src = tmp_path / "d.json"
        src.write_text(json.dumps([{"response": "r", "question": "q"}]))
        models = ["-m", "m"] if command == ["extract"] else ["-e", "e", "--checker-model", "c"]
        CliRunner().invoke(app, [*command, str(src), *models, "-o", str(tmp_path / "o.json"), *extra])
        assert seen, "the command did not build its service"
        return seen

    @pytest.mark.parametrize("command, target", [
        (["extract"], "claimlens.services.extraction.ExtractionService"),
        (["eval", "extractor"], "claimlens.eval.extractoreval.ExtractorEvaluator"),
    ])
    def test_default_and_off(self, tmp_path, monkeypatch, command, target):
        assert self._invoke(tmp_path, monkeypatch, command, target)["question_if"] == "needs_question"
        assert self._invoke(tmp_path, monkeypatch, command, target, "--question-if", "")["question_if"] == ""

    @pytest.mark.parametrize("command", ["refcheck", "ragcheck", "faithcheck", "compare"])
    def test_other_commands_have_no_option(self, command):
        from typer.testing import CliRunner
        from claimlens.cli import app
        assert "--question-if" not in CliRunner().invoke(app, [command, "--help"], terminal_width=200).output


# ── Match modes: joint / strict / many-to-many ───────────────────────────────────────
#
# The fake worker stands in for the matcher at the worker boundary, so the
# real CheckingService cuts the calls: what a test counts is what would be
# sent. A claim is entailed by a reference line that contains its text;
# subject "neg" is contradicted, subject "bad" gets no verdict, the rest is
# Neutral.

from claimlens.eval.extractoreval import MATCH_MODES, DEFAULT_MATCH_MODE  # noqa: E402
from claimlens.workers.checker import ClaimVerdict, Verdict  # noqa: E402


class _FakeChecker:
    batches: list = []

    def __init__(self, **kwargs):
        self.last_stats = None

    async def check_joint_batch(self, chunks, extra_vars_list=None, description=None):
        _FakeChecker.batches.append(chunks)
        out = []
        for numbered, reference in chunks:
            ref = "\n".join(reference)
            verdicts = {}
            for cid, text in numbered:
                if text.startswith("bad "):
                    verdicts[cid] = ClaimVerdict(verdict=None, error="parse_failure")
                elif any(line.endswith(") " + text) for line in ref.split("\n")):
                    verdicts[cid] = ClaimVerdict(verdict=Verdict.ENTAILMENT,
                                                 explanation=f"same as {ref.split(' ')[0]}")
                elif text.startswith("neg "):
                    verdicts[cid] = ClaimVerdict(verdict=Verdict.CONTRADICTION,
                                                 explanation="opposite")
                else:
                    verdicts[cid] = ClaimVerdict(verdict=Verdict.NEUTRAL, explanation="absent")
            out.append(verdicts)
        return out


def _T(s, p="p", o="o"):
    return {"subject": s, "predicate": p, "object": o}


@pytest.fixture
def fake_checker():
    _FakeChecker.batches = []
    with patch("claimlens.services.checking.Checker", _FakeChecker):
        yield _FakeChecker


def _mode_item(gt, pred, item_id="i1"):
    return _make_item(gt_triplets=[dict(t) for t in gt],
                      pred_triplets=[dict(t) for t in pred], item_id=item_id)


class TestMatchModeOption:

    def test_default_is_joint(self):
        assert DEFAULT_MATCH_MODE == "joint"
        assert _evaluator()._match_mode == "joint"

    def test_the_three_modes(self):
        assert MATCH_MODES == ("joint", "strict", "many-to-many")
        for mode in MATCH_MODES:
            assert _evaluator(match_mode=mode)._match_mode == mode

    def test_unknown_mode_fails_fast(self):
        with pytest.raises(InvalidInputError, match="match_mode"):
            _evaluator(match_mode="pairs")

    def test_list_is_no_mode(self):
        with pytest.raises(InvalidInputError, match="match_mode"):
            _evaluator(match_mode="list")


class TestCallsPerMode:
    """The calls built for one answer: which claims, which reference claim,
    how many calls. m = 3 claims judged against n = 2 reference claims in
    pass 1, the other way round in pass 2; joint_num = 2."""

    GT = [_T("g1"), _T("g2"), _T("g3")]
    PRED = [_T("p1"), _T("p2")]

    async def _passes(self, mode):
        ev = _evaluator(match_mode=mode, joint_num=2)
        await ev._match_all_llm([_mode_item(self.GT, self.PRED)])
        assert len(_FakeChecker.batches) == 2  # one batch per pass
        return _FakeChecker.batches

    async def test_many_to_many_sends_the_whole_reference_list(self, fake_checker):
        p1, p2 = await self._passes("many-to-many")
        # ⌈m / joint-num⌉ calls, each against the whole reference list
        assert [[c for c, _ in n] for n, _ in p1] == [[1, 2], [1]]
        assert [r for _, r in p1] == [["(R1) p1 p o\n(R2) p2 p o"]] * 2
        assert [[t for _, t in n] for n, _ in p1] == [["g1 p o", "g2 p o"], ["g3 p o"]]
        assert len(p2) == 1
        assert p2[0][1] == ["(R1) g1 p o\n(R2) g2 p o\n(R3) g3 p o"]

    async def test_joint_bundles_claims_against_one_reference_claim(self, fake_checker):
        p1, p2 = await self._passes("joint")
        # pass 1: n × ⌈m / joint-num⌉ = 2 × 2 calls
        assert [([t for _, t in n], r) for n, r in p1] == [
            (["g1 p o", "g2 p o"], ["(R1) p1 p o"]),
            (["g3 p o"], ["(R1) p1 p o"]),
            (["g1 p o", "g2 p o"], ["(R2) p2 p o"]),
            (["g3 p o"], ["(R2) p2 p o"]),
        ]
        # pass 2: the bundle is on the side being judged again, 3 × ⌈2 / 2⌉
        assert [([t for _, t in n], r) for n, r in p2] == [
            (["p1 p o", "p2 p o"], ["(R1) g1 p o"]),
            (["p1 p o", "p2 p o"], ["(R2) g2 p o"]),
            (["p1 p o", "p2 p o"], ["(R3) g3 p o"]),
        ]

    async def test_strict_is_one_claim_against_one_reference_claim(self, fake_checker):
        p1, p2 = await self._passes("strict")
        assert len(p1) == 3 * 2 and len(p2) == 2 * 3   # m × n per pass
        assert all(len(n) == 1 and len(r) == 1 and "\n" not in r[0] for n, r in p1 + p2)
        assert sorted((n[0][1], r[0]) for n, r in p1) == sorted(
            (f"g{j} p o", f"(R{k}) p{k} p o") for j in (1, 2, 3) for k in (1, 2))

    async def test_joint_and_strict_ask_the_same_pairs(self, fake_checker):
        def pairs(batches):
            return sorted((t, r[0]) for p in batches for n, r in p for _, t in n)
        joint = pairs(await self._passes("joint"))
        _FakeChecker.batches = []
        strict = pairs(await self._passes("strict"))
        assert joint == strict and len(joint) == 2 * 3 * 2

    @pytest.mark.parametrize("m, n, joint_num", [(1, 1, 10), (4, 3, 10), (11, 2, 10), (5, 4, 2)])
    async def test_call_counts_follow_the_formulas(self, fake_checker, m, n, joint_num):
        from math import ceil
        gt = [_T(f"g{j}") for j in range(m)]
        pred = [_T(f"p{k}") for k in range(n)]
        expected = {
            "strict": (m * n, n * m),
            "joint": (n * ceil(m / joint_num), m * ceil(n / joint_num)),
            "many-to-many": (ceil(m / joint_num), ceil(n / joint_num)),
        }
        for mode, calls in expected.items():
            _FakeChecker.batches = []
            ev = _evaluator(match_mode=mode, joint_num=joint_num)
            await ev._match_all_llm([_mode_item(gt, pred)])
            assert tuple(len(b) for b in _FakeChecker.batches) == calls, mode


class TestFoldPairs:
    """Every row of the fold table (docs/eval_extractor.md)."""

    @staticmethod
    def _pair(index, verdict, explanation="e", **extra):
        return {"index": index, "verdict": verdict, "explanation": explanation, **extra}

    def fold(self, *pairs):
        return ExtractorEvaluator._fold_pairs(list(pairs), "prediction")

    def test_one_entailment_is_found_whatever_the_others_say(self):
        cell = self.fold(self._pair(0, "Neutral"), self._pair(1, "Entailment", "R2 says it"),
                         self._pair(2, "Contradiction"))
        assert cell["verdict"] == "Entailment"
        assert cell["explanation"] == "R2 says it"
        assert cell["matched_by"] == [1]
        assert "error" not in cell

    def test_entailment_also_when_a_pair_has_no_verdict(self):
        cell = self.fold(self._pair(0, None, None, error="timeout"), self._pair(1, "Entailment"))
        assert cell["verdict"] == "Entailment" and cell["matched_by"] == [1]
        assert "error" not in cell

    def test_every_entailing_reference_claim_is_named(self):
        cell = self.fold(self._pair(0, "Entailment", "first"), self._pair(1, "Neutral"),
                         self._pair(2, "Entailment", "second"))
        assert cell["matched_by"] == [0, 2]
        assert cell["explanation"] == "first"

    def test_contradiction_when_all_judged_and_one_contradicts(self):
        cell = self.fold(self._pair(0, "Neutral"), self._pair(1, "Contradiction"))
        assert cell["verdict"] == "Contradiction"
        assert cell["explanation"] == "No single prediction makes the claim true (1 Neutral, 1 Contradiction)."
        assert cell["matched_by"] == []

    def test_neutral_when_all_judged_and_none_contradicts(self):
        cell = self.fold(self._pair(0, "Neutral"), self._pair(1, "Neutral"))
        assert cell["verdict"] == "Neutral"
        assert cell["explanation"] == "No single prediction makes the claim true (2 Neutral)."

    def test_no_verdict_when_a_pair_has_none_and_nothing_entails(self):
        cell = self.fold(self._pair(0, "Contradiction"), self._pair(1, None, None, error="timeout"))
        assert cell["verdict"] is None          # unjudged, never a miss
        assert cell["error"] == "timeout"
        assert cell["explanation"] == ("No verdict: 1 of 2 pairs returned none, "
                                       "and no other pair makes the claim true.")

    def test_no_verdict_when_every_pair_failed(self):
        cell = self.fold(self._pair(0, None, None, error="parse_failure"))
        assert cell["verdict"] is None and cell["error"] == "parse_failure"
        assert cell["explanation"] == "No verdict: no pair of the 1 returned one."

    def test_explanation_is_always_a_string(self):
        cell = self.fold(self._pair(3, "Entailment", None))
        assert cell["explanation"] == "R4 makes the claim true."

    def test_retry_follows_the_deciding_pair(self):
        found = self.fold(self._pair(0, "Neutral", retry=2), self._pair(1, "Entailment", retry=1))
        assert found["retry"] == 1
        first_pass = self.fold(self._pair(0, "Entailment"), self._pair(1, "Neutral", retry=2))
        assert "retry" not in first_pass
        # a miss is decided by every pair: each keeps its own round
        miss = self.fold(self._pair(0, "Neutral", retry=2), self._pair(1, "Neutral"))
        assert "retry" not in miss and miss["pairs"][0]["retry"] == 2

    def test_the_pairs_are_kept_in_reference_order(self):
        pairs = [self._pair(0, "Neutral"), self._pair(1, "Entailment")]
        assert self.fold(*pairs)["pairs"] == pairs


class TestRecordPerMode:
    """The record and findings of one answer, run end to end through the
    evaluator with the fake worker: cells, pairs, findings, _meta."""

    GT = [_T("a"), _T("neg"), _T("bad"), _T("c")]
    PRED = [_T("a"), _T("z")]

    def _run(self, mode, **kwargs):
        ev = _evaluator(match_mode=mode, **kwargs)

        async def extract(items):
            for item in items:
                item[PRED_KEY] = [dict(t) for t in self.PRED]
        ev._extraction_service.run = AsyncMock(side_effect=extract)
        data = [_make_item(gt_triplets=[dict(t) for t in self.GT], item_id="x")]
        with patch("claimlens.services.checking.Checker", _FakeChecker):
            return ev.run_sync(data)

    @pytest.mark.parametrize("mode", ["joint", "strict"])
    def test_cells_carry_their_pairs(self, mode):
        record, _ = self._run(mode)
        [item] = record["runs"][0]["items"]
        # parallel to the claims, as in many-to-many
        assert len(item["pred2gt"]) == len(item["gt_claims"]) == 4
        assert len(item["gt2pred"]) == len(item["pred_claims"]) == 2
        found, contradicted, unjudged, neutral = item["pred2gt"]
        assert found == {
            "verdict": "Entailment", "explanation": "same as (R1)", "matched_by": [0],
            "pairs": [{"index": 0, "verdict": "Entailment", "explanation": "same as (R1)"},
                      {"index": 1, "verdict": "Neutral", "explanation": "absent"}]}
        assert contradicted["verdict"] == "Contradiction"
        assert contradicted["explanation"] == ("No single prediction makes the claim true "
                                               "(2 Contradiction).")
        assert unjudged["verdict"] is None and unjudged["error"] == "parse_failure"
        assert neutral["verdict"] == "Neutral" and neutral["matched_by"] == []
        assert [p["index"] for p in neutral["pairs"]] == [0, 1]
        # pass 2: a prediction is judged against each ground-truth claim
        assert [len(c["pairs"]) for c in item["gt2pred"]] == [4, 4]
        assert item["gt2pred"][1]["explanation"] == ("No single ground-truth claim makes the "
                                                     "claim true (4 Neutral).")

    @pytest.mark.parametrize("mode", ["joint", "strict"])
    def test_scores_come_from_the_folded_verdicts(self, mode):
        record, _ = self._run(mode)
        counts = record["runs"][0]["counts"]
        assert counts["recall"] == {"total_gt_claims": 4, "covered": 1, "missed": 2,
                                    "answer_missed_penalty": 0, "unjudged": 1, "denominator": 3}
        assert counts["precision"]["supported"] == 1 and counts["precision"]["unsupported"] == 1
        assert counts["reliability"]["matching_checker"]["issued"] == 6

    @pytest.mark.parametrize("mode", ["joint", "strict"])
    def test_findings_show_the_pairs(self, mode):
        _, findings = self._run(mode)
        f = findings["runs"][0]["findings"]
        assert [e["claim"] for e in f["missed"]] == [_T("neg"), _T("c")]
        neg = f["missed"][0]
        assert neg["verdict"] == "Contradiction"
        assert neg["pairs"] == [
            {"index": 0, "reference_claim": _T("a"), "verdict": "Contradiction", "explanation": "opposite"},
            {"index": 1, "reference_claim": _T("z"), "verdict": "Contradiction", "explanation": "opposite"}]
        [unsupported] = f["unsupported"]
        assert unsupported["claim"] == _T("z")
        assert [p["reference_claim"] for p in unsupported["pairs"]] == self.GT
        [unjudged] = f["unjudged"]
        assert unjudged["cause"] == "parse_failure"
        assert [p["error"] for p in unjudged["pairs"]] == ["parse_failure"] * 2
        # what a finding is judged against is the same as in many-to-many
        assert neg["gt_claims"] == self.GT and neg["pred_claims"] == self.PRED

    @pytest.mark.parametrize("mode, joint_num", [("joint", 10), ("strict", 1), ("many-to-many", 10)])
    def test_meta_names_the_mode(self, mode, joint_num):
        record, findings = self._run(mode)
        for doc in (record, findings):
            assert doc["_meta"]["match_mode"] == mode
            assert doc["_meta"]["joint_num"] == joint_num
            assert doc["runs"][0]["_meta"]["match_mode"] == mode
            assert doc["_meta"]["matching"] == "llm-2-pass"

    def test_many_to_many_cells_are_the_services_own(self):
        """many-to-many: no pairs, no matched_by — every cell the service's own
        {verdict, explanation, error?}; findings without pairs."""
        record, findings = self._run("many-to-many")
        [item] = record["runs"][0]["items"]
        assert item["pred2gt"] == [
            {"verdict": "Entailment", "explanation": "same as (R1)"},
            {"verdict": "Contradiction", "explanation": "opposite"},
            {"verdict": None, "explanation": None, "error": "parse_failure"},
            {"verdict": "Neutral", "explanation": "absent"},
        ]
        assert item["gt2pred"] == [
            {"verdict": "Entailment", "explanation": "same as (R1)"},
            {"verdict": "Neutral", "explanation": "absent"},
        ]
        f = findings["runs"][0]["findings"]
        assert not any("pairs" in e for branch in f.values() for e in branch)
        assert record["runs"][0]["counts"]["recall"]["covered"] == 1

    def test_item_keys_are_the_same_in_every_mode(self):
        """An item has the same keys in every mode, and every cell of
        pred2gt and gt2pred keeps verdict and explanation."""
        item_keys = {"id", "question", "response", "bucket", "gt_claims", "pred_claims",
                     "pred2gt", "gt2pred"}
        for mode in MATCH_MODES:
            record, _ = self._run(mode)
            [item] = record["runs"][0]["items"]
            assert set(item) == item_keys
            for cells in (item["pred2gt"], item["gt2pred"]):
                assert all({"verdict", "explanation"} <= set(c) for c in cells)
                assert all(isinstance(c["explanation"], str) for c in cells
                           if mode != "many-to-many")

    def test_config_prints_the_mode(self, caplog):
        import logging
        for mode, line in [
                ("many-to-many", "    Match mode:  many-to-many · up to 10 claims against the "
                                 "whole list of reference claims per call · 6000 max words\n"),
                ("joint", "    Match mode:  joint · up to 10 claims against one reference claim "
                          "per call · 6000 max words\n"),
                ("strict", "    Match mode:  strict · one claim against one reference claim "
                           "per call\n")]:
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="claimlens"):
                _evaluator(match_mode=mode)._log_eval_config([])
            assert line in caplog.text + "\n", mode
            # "joint" names the match mode only: no service Mode: line
            assert "    Mode:" not in caplog.text, mode
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="claimlens"):
            _evaluator(match_mode="joint", max_words=3000)._log_eval_config([])
        assert "per call · 3000 max words" in caplog.text


class TestMatchModeCli:

    def _invoke(self, tmp_path, monkeypatch, *extra):
        import json
        from typer.testing import CliRunner
        from claimlens.cli import app
        seen = {}

        class _Built:
            def __init__(self, **kwargs):
                seen.update(kwargs)

            def run_sync(self, data):
                return {"_meta": {}}, {"_meta": {}}

        monkeypatch.setattr("claimlens.eval.extractoreval.ExtractorEvaluator", _Built)
        src = tmp_path / "d.json"
        src.write_text(json.dumps([{"response": "r"}]))
        CliRunner().invoke(app, ["eval", "extractor", str(src), "-e", "e", "--checker-model",
                                 "c", "-o", str(tmp_path / "o.json"), *extra])
        return seen

    def test_default_is_joint(self, tmp_path, monkeypatch):
        assert self._invoke(tmp_path, monkeypatch)["match_mode"] == "joint"

    @pytest.mark.parametrize("mode", ["joint", "strict", "many-to-many"])
    def test_passed_through(self, tmp_path, monkeypatch, mode):
        assert self._invoke(tmp_path, monkeypatch, "--match-mode", mode)["match_mode"] == mode

    def test_unknown_mode_exits_with_an_error(self, tmp_path):
        import json
        from typer.testing import CliRunner
        from claimlens.cli import app
        src = tmp_path / "d.json"
        src.write_text(json.dumps([{"response": "r", "claude2_response_kg": []}]))
        result = CliRunner().invoke(app, ["eval", "extractor", str(src), "-e", "e",
                                          "--checker-model", "c", "--match-mode", "pairs",
                                          "-o", str(tmp_path / "o.json")])
        assert result.exit_code == 1
        assert not (tmp_path / "o.json").exists()

    @pytest.mark.parametrize("command", ["check", "refcheck", "ragcheck", "faithcheck",
                                         "compare", ["eval", "checker"]])
    def test_other_commands_have_no_option(self, command):
        from typer.testing import CliRunner
        from claimlens.cli import app
        args = command if isinstance(command, list) else [command]
        assert "--match-mode" not in CliRunner().invoke(
            app, [*args, "--help"], terminal_width=200).output
