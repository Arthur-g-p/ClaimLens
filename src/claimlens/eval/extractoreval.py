"""
Extractor Evaluator — triplet-level extraction quality measurement.

Runs extraction live on GT-annotated data, then compares predicted
triplets against ground-truth using set-to-set IR matching.

Architecture:
    - Delegates extraction to ExtractionService (quiet — evaluator owns logging).
    - Matching mode: LLM (online, 2-pass checker).
    - Strict separation: _validate (fail-fast) → extract → _classify (bucket sort).
    - Owns: GT validation, matching orchestration, IR metric computation,
      the record + findings documents, and the ── EXTRACTOR EVAL ── logging section.
    - Does NOT inherit BaseService — evaluators measure, services mutate.
"""

import copy
import time
from dataclasses import dataclass, field
from datetime import datetime

from claimlens import prompts, settings
from claimlens.exceptions import InvalidInputError
from claimlens.models import DEFAULT_RETRY_ROUNDS, ExtractorEvalResult
from claimlens.stats import (
    GLOBAL_STATS,
    usage_since,
    log_mece_tree,
    format_headline,
    log_rate_rows,
)
from claimlens.utils import (
    plural,
    build_meta,
    canonicalize_triplets,
    describe_retry_rounds,
    find_duplicate_triplets,
    findings_view,
    retry_of,
)
from claimlens.eval.base import Evaluator
from claimlens.pipelines.directions import _flat_cell, _spo
from claimlens.services.base import BaseService
from claimlens.services.checking import CheckingService
from claimlens.services.extraction import ExtractionService
from claimlens.services.atomization import AtomizationService

logger = settings.get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

# Internal model name for the checking service's kg_key.
_INTERNAL_EXT_MODEL = "_exteval"

# How one matcher call is cut (docs/eval_extractor.md, "The three modes"):
#   joint        — up to joint_num claims against one reference claim
#   strict       — one claim against one reference claim
#   many-to-many — up to joint_num claims against the whole list of reference claims
# joint and strict ask the same pairs, one-to-one; only many-to-many lets the
# matcher see several reference claims at once.
MATCH_MODES = ("joint", "strict", "many-to-many")
DEFAULT_MATCH_MODE = "joint"
_PAIR_MODES = ("joint", "strict")

_ENTAILMENT = "Entailment"
_CONTRADICTION = "Contradiction"


# ── Item classification ─────────────────────────────────────────────────────

@dataclass
class _ItemBucket:
    """Post-extraction classification into five buckets.

    Abstention naming follows the ragcheck pipeline: an abstention is
    Two response kinds × two extraction outcomes: an answering response is
    extracted or missed entirely; an abstaining response (GT has no claims)
    is recognized or misread into invented claims.
    """
    to_compare: list[dict]              # has GT + has predictions → normal matching
    abstention_misread: list[dict]     # abstaining response, claims invented → precision penalty
    answer_missed: list[dict]          # answering response, nothing extracted → recall penalty
    abstention_recognized: list[dict]  # abstaining response, nothing extracted → correct
    # extraction failed (tooling) → excluded from ALL metrics, counted in
    # the error rate
    extraction_error: list[dict] = field(default_factory=list)


# ── Matching result per item ─────────────────────────────────────────────────

@dataclass
class _ItemMatchResult:
    """Matching outcome for a single valid item.

    RAGChecker uses two independent coverage counts (no shared TP / no min):
      - tp_recall:    GT triplets entailed by the predictions  (recall numerator)
      - tp_precision: pred triplets entailed by the GT         (precision numerator)

    fp/fn count judged misses only. Claims the checker never returned a
    verdict for (tooling failure) sit in unjudged_* — excluded from both
    numerator and denominator, never charged to the extractor.
    """
    tp_recall: int
    tp_precision: int
    fp: int
    fn: int
    false_positives: list[dict]     # judged misses, pass 2 (one per FP)
    false_negatives: list[dict]     # judged misses, pass 1 (one per FN)
    unjudged_gt: list[dict] = field(default_factory=list)    # pass 1 checker failures
    unjudged_pred: list[dict] = field(default_factory=list)  # pass 2 checker failures
    # The complete record, ragcheck's shape: one verdict cell per claim,
    # parallel to the item's GT claims (pass 1) and predictions (pass 2).
    pred2gt: list[dict] = field(default_factory=list)
    gt2pred: list[dict] = field(default_factory=list)


def _bucket_rate(count: int, denominator: int) -> float | None:
    """Rate over the behavioral item universe; None on an empty universe."""
    return round(count / denominator, 4) if denominator else None


class ExtractorEvaluator(Evaluator):
    """Evaluates extraction quality by extracting live then matching against GT.

    Flow: validate → extract (API) → classify → match (LLM) → metrics.

    Matching mode:
        - LLM: 2-pass batched checker with semantic equivalence prompt.
          Pass 1: GT→Pred (recall). Pass 2: Pred→GT (precision).
        - match_mode cuts each call: ``joint`` (claims against one reference
          claim), ``strict`` (one claim against one reference claim),
          ``many-to-many`` (claims against the whole reference list). In joint
          and strict every pair gets its own verdict, folded to one per claim.

    Returns (record, findings) — CLI writes two files verbatim.
    """

    _RUN_SUMMARY_KEYS = ("recall", "precision", "f1")

    _VARIANCE_SECTIONS = {
        "metrics": [
            ("Matching Quality", ["recall", "precision", "f1"]),
            ("Atomicity", ["atomicity_rate", "claim_density"]),
            ("Duplicates", ["duplicate_rate"]),
        ],
        "behavior": ["abstention_recognized_rate", "abstention_misread_rate",
                     "answer_missed_rate"],
        "behavior_title": "Abstention Handling",
        "health": ["extraction_error_rate", "checker_failure_rate",
                   "atomization_failure_rate"],
    }
    _METRIC_DIRECTIONS = {
        "recall": "higher is better", "precision": "higher is better", "f1": "higher is better",
        "atomicity_rate": "higher is better",
        "claim_density": "closer to 1.0 is better",
        "duplicate_rate": "lower is better",
        "abstention_recognized_rate": "higher is better",
        "abstention_misread_rate": "lower is better",
        "answer_missed_rate": "lower is better",
    }
    # Per-run lines that name their key (`→ atomicity_rate 1.000`) keep the
    # key in the variance block — same name, rule set 1.1 / 2.2.
    _VARIANCE_LABELS = {
        "atomicity_rate": "atomicity_rate",
        "claim_density": "claim_density",
        "duplicate_rate": "duplicate_rate",
    }

    def __init__(
        self,
        extractor_model: str,
        gt_key: str = "claude2_response_kg",
        # Extraction config
        extractor_base_url: str | None = None,
        # LLM matching config
        checker_model: str | None = None,
        checker_base_url: str | None = None,
        concurrency: int = 10,
        prompt_cache: bool = True,
        templates: prompts.PromptMap | None = None,
        joint_num: int = settings.DEFAULT_JOINT_NUM,
        max_words: int | None = None,
        # How one matcher call is cut: "joint" | "strict" | "many-to-many" (MATCH_MODES)
        match_mode: str = DEFAULT_MATCH_MODE,
        # Atomicity axis (optional, orthogonal to coverage)
        atomizer_model: str | None = None,
        atomizer_base_url: str | None = None,
        runs: int = 1,
        # Items carrying this key (truthy) are left out: not extracted, not matched
        skip_marked: str | None = None,
        # Items whose value under this key is true are extracted with their
        # question (the question prompt); None / "" = off
        question_if: str | None = None,
    ):
        if not checker_model:
            raise ValueError("checker_model is required for evaluation matching.")
        if match_mode not in MATCH_MODES:
            raise InvalidInputError(
                f"match_mode must be one of {', '.join(MATCH_MODES)}, got '{match_mode}'."
            )

        self._extractor_model = extractor_model
        self._gt_key = gt_key
        self._skip_marked = skip_marked
        self._question_if = question_if or None
        # runs > 1 = variance mode: the evaluator repeats itself; the
        # controller only passes the number through.
        self._runs = max(1, runs)
        # Predictions are always written by ExtractionService(model=extractor_model)
        # under this key — it is derived, never an override.
        self._pred_key = f"{extractor_model}_response_kg"
        self._error_key = f"{extractor_model}_extraction_error"

        # Guard: the predicted-triplet key must not collide with the GT key.
        # On collision (e.g. extractor_model="claude2" with the default gt_key)
        # extraction would target the GT slot and the evaluator would match
        # ground truth against itself, silently reporting perfect P/R/F1.
        if self._pred_key == self._gt_key:
            raise InvalidInputError(
                f"Predicted-triplet key collides with GT key: '{self._gt_key}'. "
                f"The predicted key is derived as '{{extractor_model}}_response_kg' "
                f"(here '{self._pred_key}'). Use a different --extractor-model or "
                f"--gt-key so the two do not overlap."
            )

        self._templates = templates or prompts.PROMPTS  # resolved at the entry point, passed down
        # Extraction service — always needed (quiet, evaluator owns logging;
        # silent in variance mode — per-run plumbing is cut).
        # dedup=False: the eval measures duplicates as an independent dimension,
        # so predictions must stay raw — we never evaluate on deduplicated data.
        self._extraction_service = ExtractionService(
            model=extractor_model,
            base_url=extractor_base_url,
            concurrency=concurrency,
            prompt_cache=prompt_cache,
            templates=self._templates,
            verbosity="silent" if self._runs > 1 else "compact",
            dedup=False,
            question_if=self._question_if,
        )

        # LLM config — CheckingService built on demand in _match_all_llm
        self._checker_model = checker_model
        self._checker_base_url = checker_base_url
        self._concurrency = concurrency
        self._prompt_cache = prompt_cache
        self._joint_num = joint_num
        self._max_words = max_words
        self._match_mode = match_mode

        # Atomicity axis — OPTIONAL. Built only if an atomizer model is given AND
        # the key is configured (AtomizationService.__init__ would otherwise raise
        # on a missing key). Measured on a throwaway copy of the predictions, so
        # coverage numbers and the output file are never affected.
        self._atomizer_model = atomizer_model
        self._atomization_service = None
        self._atomizer_skip_reason = None
        if not atomizer_model:
            self._atomizer_skip_reason = "no --atomizer-model"
        elif not settings.ATOMIZER_API_KEY:
            self._atomizer_skip_reason = "ATOMIZER_API_KEY not set"
        else:
            self._atomization_service = AtomizationService(
                model=atomizer_model,
                source_kg_key=self._pred_key,
                base_url=atomizer_base_url,
                concurrency=concurrency,
                prompt_cache=prompt_cache,
                templates=self._templates,
                verbosity="silent" if self._runs > 1 else "compact",
            )

    # ── Public API ───────────────────────────────────────────────

    async def _evaluate_once(
        self, data: list[dict], announce: bool = True,
    ) -> tuple[dict, dict]:
        """Run the full extractor evaluation pipeline once.

        Args:
            data: Pre-loaded list of items (GT-annotated dataset with response text).
            announce: Print the Data/Config sections (False for runs 2..N
                in variance mode — they repeat run 1 verbatim).

        Returns:
            (run_doc, run_findings): one ``runs[]`` entry of the record —
            {_meta, metrics, counts, items} — and the matching findings
            entry {_meta, findings}, derived from the items.
        """
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._started_perf = time.perf_counter()
        self._usage_at_start = GLOBAL_STATS.snapshot()

        # Step 0: Canonicalize key aliases
        BaseService._canonicalize_keys(data)

        # Step 1: Validate — fail-fast on bad input
        valid = self._validate(data)

        for item in valid:
            if item.get(self._gt_key):
                canonicalize_triplets(item[self._gt_key])

        # Pre-extraction logging
        if announce:
            self._log_data_pre(len(data), len(data) - len(valid), len(valid))
            self._log_eval_config(valid)

        # Step 2: Run extraction (quiet — evaluator owns the logging)
        await self._extraction_service.run(valid)

        for item in valid:
            if item.get(self._pred_key):
                canonicalize_triplets(item[self._pred_key])

        # Step 2b: Atomicity axis (optional, orthogonal). Measured on a COPY so
        # coverage and the output file keep the raw predictions untouched.
        atomicity = await self._measure_atomicity(valid)

        # Step 2c: Duplicate-claims axis (orthogonal, read-only). The eval never
        # evaluates on deduplicated data — it only reports the duplicates.
        duplicates = self._measure_duplicates(valid)

        # Step 3: Classify into buckets (now that pred exists)
        buckets = self._classify(valid)

        # Step 4: Guard — need at least one to_compare item to match
        if not buckets.to_compare:
            logger.warning(
                "No items with both GT and predictions after extraction. "
                "All %d items were answers with nothing extracted.",
                len(buckets.answer_missed),
            )

        # Step 5: Match to_compare items (LLM)
        if buckets.to_compare:
            item_results = await self._match_all_llm(buckets.to_compare)
        else:
            item_results = []

        # Step 6: Build result (encapsulated)
        result = self._build_result(item_results, buckets, len(data), atomicity, duplicates)

        # Step 7: Log eval results (findings blocks). Done line and token
        # table belong to the base loop — they print once per invocation.
        self._log_eval_results(result)

        # Step 8: Assemble this run's documents. The CLI only resolves
        # paths and dumps JSON — it never composes content.
        return self._run_documents(result, buckets, item_results, len(data))

    @property
    def _bundle_size(self) -> int:
        """Claims per matcher call as the run uses it: one in strict,
        joint_num in joint and many-to-many."""
        return 1 if self._match_mode == "strict" else self._joint_num

    def _match_mode_line(self) -> str:
        """The Config "Match mode:" string: what one matcher call holds, and
        the word budget where a bundle can shrink (joint, many-to-many), resolved as
        the service resolves it."""
        if self._match_mode == "strict":
            return "strict · one claim against one reference claim per call"
        words = (self._max_words if self._max_words is not None
                 else settings.DEFAULT_MAX_WORDS)
        if self._match_mode == "joint":
            return (f"joint · up to {self._joint_num} claims against one "
                    f"reference claim per call · {words} max words")
        return (f"many-to-many · up to {self._joint_num} claims against the whole "
                f"list of reference claims per call · {words} max words")

    def _gets_question(self, item: dict) -> bool:
        """Extracted with its question? The same rule as ExtractionService.is_marked:
        only the value true counts."""
        return self._question_if is not None and item.get(self._question_if) is True

    def _unmeasured(self) -> dict | None:
        """Atomicity axis not run → its three keys are unmeasured, not 0/0."""
        if not self._atomizer_skip_reason:
            return None
        return {k: self._atomizer_skip_reason for k in
                ("atomicity_rate", "claim_density", "atomization_failure_rate")}

    # ── Output documents (serialization owned by the evaluator) ──

    def _run_documents(
        self,
        result: ExtractorEvalResult,
        buckets: _ItemBucket,
        item_results: list[_ItemMatchResult],
        total_items: int,
    ) -> tuple[dict, dict]:
        """One ``runs[]`` entry of the record plus its findings entry.

        ``metrics`` are the varianced scalars (the console's Metrics roster),
        ``counts`` every number the console blocks print — rates live in
        metrics only — and ``items`` the complete per-item record, one entry
        per attempted item in every bucket. Findings derive from the items.
        """
        evaluated = (len(buckets.to_compare) + len(buckets.abstention_misread)
                     + len(buckets.answer_missed) + len(buckets.abstention_recognized)
                     + len(buckets.extraction_error))
        meta = build_meta(
            "extractor_eval",
            timestamp=self._started_at,
            duration_seconds=time.perf_counter() - self._started_perf,
            total_items=total_items,
            evaluated_items=evaluated,
            dropped_items=total_items - evaluated,
            pred_key=self._pred_key,
            matching="llm-2-pass",
            match_mode=self._match_mode,
            joint_num=self._bundle_size,
            request_strategies=GLOBAL_STATS.strategies(),
            prompt_cache_marker=GLOBAL_STATS.cache_markers(),
            prompts_sha256=prompts.prompts_sha256(self._templates),
            prompts_overridden=prompts.prompts_overridden(self._templates),
            retry_rounds=describe_retry_rounds(DEFAULT_RETRY_ROUNDS),
            usage=usage_since(getattr(self, "_usage_at_start", None)),
        )
        metrics = {
            "recall": result.recall,
            "precision": result.precision,
            "f1": result.f1,
            "atomicity_rate": result.atomicity_rate,
            "claim_density": result.claim_density,
            "duplicate_rate": result.duplicate_rate,
            "abstention_recognized_rate": result.abstention_recognized_rate,
            "abstention_misread_rate": result.abstention_misread_rate,
            "answer_missed_rate": result.answer_missed_rate,
            "extraction_error_rate": result.extraction_error_rate,
            "checker_failure_rate": result.checker_failure_rate,
            "atomization_failure_rate": result.atomization_failure_rate,
        }

        ee = result.extraction_errors or {"count": 0, "by_cause": {}}
        cf = result.checker_failures
        a = result.atomicity
        d = result.duplicates
        atomization = ({"measured": True, "failed": a["failed"],
                        "claims": a["extracted_claims"]} if a else
                       {"measured": False,
                        "reason": self._atomizer_skip_reason or "no predictions to measure"})
        counts = {
            # 📂 Data
            "data": {
                "skipped_marked": getattr(self, "_skipped_marked", 0),
                "dropped_no_response": getattr(self, "_dropped_missing_response", 0),
                "dropped_no_gt_key": getattr(self, "_dropped_missing_gt", 0),
                "dropped_no_question": getattr(self, "_dropped_missing_question", 0),
                "gt_empty": getattr(self, "_gt_empty", 0),
            },
            # 🔎 Matching Quality — the two funnels
            "recall": result.recall_counts,
            "precision": result.precision_counts,
            # 📊 Extraction Stats
            "extraction_stats": {
                "gt": {"claims": result.gt_stats["total_triplets"],
                       "avg_per_item": result.gt_stats["avg_per_item"]},
                "pred": {"claims": result.pred_stats["total_triplets"],
                         "avg_per_item": result.pred_stats["avg_per_item"]},
            },
            # ⚪ Abstention Handling
            "abstention_handling": result.abstention_handling,
            # 💥 Reliability — one entry per row
            "reliability": {
                "extraction": {"failed": ee["count"], "items": evaluated,
                               "by_cause": ee.get("by_cause") or {}},
                "matching_checker": {
                    "unjudged": cf["count"], "issued": cf["issued_verdicts"],
                    "items_affected": cf["items_affected"],
                    "unjudged_gt": cf["unjudged_gt"],
                    "unjudged_pred": cf["unjudged_pred"],
                },
                "atomization": atomization,
            },
            # 🧬 Atomicity (numbers only; splits ride on their items)
            "atomicity": ({k: a[k] for k in ("extracted_claims", "evaluated_claims",
                                             "atomic_units", "new_claims_from_splits",
                                             "non_atomic")} if a else None),
            # 🔁 Duplicates (numbers only; the duplicates ride on their items)
            "duplicates": ({k: d[k] for k in ("predicted_claims", "unique_claims",
                                              "duplicate_claims")} if d else None),
        }
        items = self._build_items(buckets, item_results)
        self._attach_per_item_detail(items, a, d)
        run_doc = {"_meta": meta, "metrics": metrics, "counts": counts,
                   "items": items}
        run_findings = {"_meta": dict(meta), "findings": self._build_findings(items)}
        return run_doc, run_findings

    @staticmethod
    def _attach_per_item_detail(items: list[dict], atomicity: dict | None,
                                duplicates: dict | None) -> None:
        """Per-item detail of the orthogonal axes — split decisions and
        duplicate claims — lands on the item it belongs to (matched by id)."""
        by_id = {item["id"]: item for item in items}
        for split in (atomicity or {}).get("splits", []):
            item = by_id.get(split.get("id"))
            if item is not None:
                item.setdefault("atomicity_splits", []).append(
                    {k: split[k] for k in ("original", "children", "reasoning")})
        for entry in (duplicates or {}).get("items", []):
            item = by_id.get(entry.get("id"))
            if item is not None:
                item["duplicate_claims"] = entry["duplicates"]

    # ── Atomicity axis (optional, orthogonal to coverage) ────────

    async def _measure_atomicity(self, valid: list[dict]) -> dict | None:
        """Measure how atomic the extractor's predictions are.

        Runs the atomizer on a DEEP COPY of the items so the predictions used
        for coverage matching (and the output file) are never mutated — this is
        a pure measurement side-channel. Skipped (returns None) when no atomizer
        is configured.
        """
        if self._atomization_service is None:
            logger.info(
                " ⏭️  Atomicity skipped (%s)", self._atomizer_skip_reason
            )
            return None

        # Nothing to measure if no item carries predictions yet.
        if not any(item.get(self._pred_key) for item in valid):
            logger.info(" ⏭️  Atomicity skipped (no predictions to measure)")
            return None

        sandbox = copy.deepcopy(valid)
        await self._atomization_service.run(sandbox)
        trace = self._atomization_service.last_trace

        keep = split = failed = atomic_units = 0
        splits: list[dict] = []
        for item in trace:
            for d in item.get("decisions", []):
                dec = d.get("decision")
                if dec == "split":
                    split += 1
                    atomic_units += len(d.get("children", []))
                    # Full split detail — attached to its item in the record.
                    orig = d.get("original", {})
                    splits.append({
                        "id": item.get("id"),
                        "original": self._triplet_to_str(orig) if orig else None,
                        "children": [
                            self._triplet_to_str(c) for c in d.get("children", [])
                        ],
                        "reasoning": d.get("reasoning"),
                    })
                elif dec == "failed":
                    failed += 1
                else:
                    keep += 1
                    atomic_units += 1

        extracted = keep + split + failed
        evaluated = keep + split
        rate = keep / evaluated if evaluated else 1.0
        density = atomic_units / evaluated if evaluated else 0.0
        return {
            "extracted_claims": extracted,
            "evaluated_claims": evaluated,
            "atomic_units": atomic_units,
            "new_claims_from_splits": atomic_units - evaluated,
            "non_atomic": split,
            "failed": failed,
            "atomicity_rate": round(rate, 4),
            "information_density": round(density, 2),
            "splits": splits,
        }

    # ── Duplicate-claims axis (orthogonal, read-only) ────────────

    def _measure_duplicates(self, valid: list[dict]) -> dict | None:
        """Count exact (by string) duplicate claims in the predictions.

        Fully independent of coverage: read-only, never mutates predictions, the
        eval is never run on deduplicated data. The duplicate triplets are listed
        per item so they land in the results file. Returns None when no item
        carries predictions.
        """
        if not any(item.get(self._pred_key) for item in valid):
            return None

        predicted = duplicate = 0
        items_out: list[dict] = []
        for item in valid:
            preds = item.get(self._pred_key) or []
            predicted += len(preds)
            dups = find_duplicate_triplets(preds)
            if dups:
                duplicate += len(dups)
                items_out.append({
                    "id": item.get("id"),
                    "duplicates": [self._triplet_to_str(t) for t in dups],
                })

        rate = duplicate / predicted if predicted else 0.0
        return {
            "predicted_claims": predicted,
            "unique_claims": predicted - duplicate,
            "duplicate_claims": duplicate,
            "duplicate_rate": round(rate, 4),
            "items": items_out,
        }

    # ── Validation (fail-fast) ───────────────────────────────────

    def _validate(self, data: list[dict]) -> list[dict]:
        """Fail-fast validation: an item needs a response to extract from
        and a GT key to be judged against. With ``skip_marked``, an item that
        carries that key is left out first: it stays in the file, marked, but
        is neither extracted nor matched (e.g. a response cut as out of scope).

        Absent or null GT key = missing data → dropped (the same rule as
        ragcheck's gt_answer: absent is missing, an explicit empty is data).
        An empty GT list is data: "nothing to extract" — the response
        abstained, which is what makes a misread abstention detectable.
        If zero survive → InvalidInputError; if every response-bearing item
        lacks the GT key, that is a wrong file or wrong --gt-key — fatal,
        and caught before any LLM call.
        """
        valid = []
        self._skipped_marked = 0
        self._dropped_missing_response = 0
        self._dropped_missing_question = 0
        self._dropped_missing_gt = 0

        for item in data:
            if self._skip_marked and item.get(self._skip_marked):
                self._skipped_marked += 1
                continue
            if not item.get("response"):
                self._dropped_missing_response += 1
                continue
            if self._gets_question(item) and not str(item.get("question") or "").strip():
                self._dropped_missing_question += 1
                continue
            if not isinstance(item.get(self._gt_key), list):
                self._dropped_missing_gt += 1
                continue
            valid.append(item)
        self._gt_empty = sum(1 for item in valid if not item[self._gt_key])

        if not valid:
            with_response = (len(data) - self._skipped_marked - self._dropped_missing_response
                             - self._dropped_missing_question)
            if with_response and self._dropped_missing_gt == with_response:
                raise InvalidInputError(
                    f"No ground truth found: none of the {with_response} "
                    f"evaluable items contain the key '{self._gt_key}'. "
                    f"`eval extractor` measures the extractor against labeled "
                    f"GT — check the input file, or point --gt-key at the "
                    f"right key."
                )
            raise InvalidInputError(
                f"No evaluable items. All {len(data)} items dropped "
                f"(skipped_marked={self._skipped_marked}, "
                f"missing_response={self._dropped_missing_response}, "
                f"missing_question={self._dropped_missing_question}, "
                f"missing_gt={self._dropped_missing_gt})."
            )

        return valid

    # ── Classification (post-extraction) ─────────────────────────

    def _classify(self, data: list[dict]) -> _ItemBucket:
        """Post-extraction classification into five buckets.

        Called AFTER ExtractionService.run() — pred_key may or may not
        be populated per item depending on extraction results. Tooling
        failures get their own bucket before any abstention judgement:
        an empty prediction list after a crashed extraction says nothing
        about the extractor's willingness to answer.
        """
        to_compare, abstention_misread = [], []
        answer_missed, abstention_recognized = [], []
        extraction_error = []

        for item in data:
            # Tooling failure — empty predictions here mean NOTHING about the
            # extractor's abstention behavior. Own bucket, out of all metrics.
            if self._error_key in item:
                extraction_error.append(item)
                continue

            has_gt = bool(item.get(self._gt_key))
            has_pred = bool(item.get(self._pred_key))

            if has_gt and has_pred:
                to_compare.append(item)
            elif not has_gt and has_pred:
                abstention_misread.append(item)
            elif has_gt and not has_pred:
                answer_missed.append(item)
            else:
                abstention_recognized.append(item)

        return _ItemBucket(
            to_compare=to_compare,
            abstention_misread=abstention_misread,
            answer_missed=answer_missed,
            abstention_recognized=abstention_recognized,
            extraction_error=extraction_error,
        )

    # ── Result building (encapsulated) ───────────────────────────

    def _build_result(
        self,
        item_results: list[_ItemMatchResult],
        buckets: _ItemBucket,
        total_items: int,
        atomicity: dict | None = None,
        duplicates: dict | None = None,
    ) -> ExtractorEvalResult:
        """Aggregate judged counts + abstention penalties, compute P/R/F1.

        RAGChecker semantics:
            recall    = entailed_GT   / total_GT      (coverage of ground truth)
            precision = entailed_pred / total_pred    (correctness of predictions)
        Two independent ratios — no shared TP, no min().

        Denominators carry judged claims + penalties. Unjudged claims
        (checker failure) leave numerator AND denominator; an empty
        denominator yields None, never 0.0 — nothing judged is not a score.
        """
        # Accumulate the judged counts + unjudged tallies from matched items
        covered = sum(ir.tp_recall for ir in item_results)
        missed = sum(ir.fn for ir in item_results)
        supported = sum(ir.tp_precision for ir in item_results)
        unsupported = sum(ir.fp for ir in item_results)
        unjudged_gt = sum(len(ir.unjudged_gt) for ir in item_results)
        unjudged_pred = sum(len(ir.unjudged_pred) for ir in item_results)

        # Answer missed: every GT triplet is an uncovered GT claim (FN)
        answer_missed_penalty = sum(
            len(item[self._gt_key]) for item in buckets.answer_missed
        )

        # Abstention misread: every predicted triplet is an unsupported prediction (FP)
        abstention_misread_penalty = sum(
            len(item[self._pred_key]) for item in buckets.abstention_misread
        )

        recall_den = covered + missed + answer_missed_penalty
        precision_den = supported + unsupported + abstention_misread_penalty

        recall = round(covered / recall_den, 4) if recall_den > 0 else None
        precision = round(supported / precision_den, 4) if precision_den > 0 else None
        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = round(2 * precision * recall / (precision + recall), 4)

        # Exhaustive partitions: total = judged + penalty + unjudged.
        recall_counts = {
            "total_gt_claims": recall_den + unjudged_gt,
            "covered": covered,
            "missed": missed,
            "answer_missed_penalty": answer_missed_penalty,
            "unjudged": unjudged_gt,
            "denominator": recall_den,
        }
        precision_counts = {
            "total_pred_claims": precision_den + unjudged_pred,
            "supported": supported,
            "unsupported": unsupported,
            "abstention_misread_penalty": abstention_misread_penalty,
            "unjudged": unjudged_pred,
            "denominator": precision_den,
        }

        # Checker failures — eval tooling reliability, co-equal with the
        # extraction error rate but never mixed into P/R/F1. Denominator:
        # every verdict actually asked of the checker (penalty claims were
        # never sent, so they don't count as issued).
        issued = covered + missed + supported + unsupported + unjudged_gt + unjudged_pred
        unjudged_total = unjudged_gt + unjudged_pred
        checker_failures = {
            "count": unjudged_total,
            "issued_verdicts": issued,
            "rate": round(unjudged_total / issued, 4) if issued else 0.0,
            "items_affected": sum(
                1 for ir in item_results if ir.unjudged_gt or ir.unjudged_pred
            ),
            "unjudged_gt": unjudged_gt,
            "unjudged_pred": unjudged_pred,
        }

        # Extraction stats
        gt_triplets = sum(
            len(item[self._gt_key])
            for item in buckets.to_compare + buckets.answer_missed
        )
        pred_triplets = sum(
            len(item[self._pred_key])
            for item in buckets.to_compare + buckets.abstention_misread
        )
        to_compare_count = len(buckets.to_compare)
        gt_avg = gt_triplets / to_compare_count if to_compare_count > 0 else 0.0
        pred_avg = pred_triplets / to_compare_count if to_compare_count > 0 else 0.0

        error_items = buckets.extraction_error
        attempted = (
            to_compare_count + len(buckets.abstention_misread)
            + len(buckets.answer_missed) + len(buckets.abstention_recognized)
            + len(error_items)
        )
        by_cause: dict[str, int] = {}
        for item in error_items:
            cause = item.get(self._error_key, "unknown")
            by_cause[cause] = by_cause.get(cause, 0) + 1
        extraction_errors = {
            "count": len(error_items),
            "rate": round(len(error_items) / attempted, 4) if attempted else 0.0,
            "by_cause": by_cause,
        }

        behavioral = attempted - len(error_items)
        unanswerable = (len(buckets.abstention_recognized)
                        + len(buckets.abstention_misread))
        answerable = to_compare_count + len(buckets.answer_missed)

        return ExtractorEvalResult(
            precision=precision,
            recall=recall,
            f1=f1,
            recall_counts=recall_counts,
            precision_counts=precision_counts,
            total_items=total_items,
            to_compare_items=to_compare_count,
            gt_stats={
                "total_triplets": gt_triplets,
                "avg_per_item": round(gt_avg, 2),
            },
            pred_stats={
                "total_triplets": pred_triplets,
                "avg_per_item": round(pred_avg, 2),
            },
            abstention_handling={
                "answers": answerable,
                "answers_extracted": to_compare_count,
                "answers_missed": len(buckets.answer_missed),
                "abstentions": unanswerable,
                "abstentions_recognized": len(buckets.abstention_recognized),
                "abstentions_misread": len(buckets.abstention_misread),
            },
            checker_failures=checker_failures,
            atomicity=atomicity,
            duplicates=duplicates,
            extraction_errors=extraction_errors,
            abstention_recognized_rate=_bucket_rate(
                len(buckets.abstention_recognized), unanswerable),
            answer_missed_rate=_bucket_rate(
                len(buckets.answer_missed), answerable),
            abstention_misread_rate=_bucket_rate(
                len(buckets.abstention_misread), unanswerable),
            atomicity_rate=atomicity["atomicity_rate"] if atomicity else None,
            claim_density=(atomicity["information_density"]
                           if atomicity else None),
            atomization_failure_rate=(
                round(atomicity["failed"] / atomicity["extracted_claims"], 4)
                if atomicity and atomicity["extracted_claims"] else None
            ),
            duplicate_rate=duplicates["duplicate_rate"] if duplicates else None,
            extraction_error_rate=extraction_errors["rate"],
            checker_failure_rate=checker_failures["rate"],
        )

    # ── LLM matching (2-pass) ────────────────────────────────────

    async def _match_all_llm(
        self, valid_items: list[dict]
    ) -> list[_ItemMatchResult]:
        """Run LLM 2-pass matching across all valid items.

        Pass 1 (GT → Pred): GT triplets as claims, pred triplets as reference.
            Entailment = extractor found this GT claim (contributes to recall).
        Pass 2 (Pred → GT): Pred triplets as claims, GT triplets as reference.
            Non-Entailment = hallucinated/wrong prediction (contributes to FP).

        The match mode decides what one call holds (MATCH_MODES). In
        ``many-to-many`` a synthetic item carries the whole reference list and
        every claim gets one verdict. In ``joint`` and ``strict`` a synthetic item
        carries one reference claim, so every (claim, reference claim) pair
        gets its own verdict; the pairs are folded to one cell per claim
        (``_fold_pairs``) before counting. The service bundles up to
        joint_num claims per call in ``joint``, one in ``strict``.

        Uses claim_match_prompt. The response text is withheld on
        purpose: given it, the matcher matched claims that had no counterpart
        on the other side (docs/eval_extractor.md, "Why the matcher never
        sees the response").
        """
        # Build a CheckingService with the eval prompt. Each pass is one
        # phase with its own labeled section rule, like a pipeline direction.
        # joint=True in every mode: joint=False would switch the worker to
        # checker_prompt, another prompt. strict bundles one claim per call.
        # In joint/strict a verdict of the service belongs to a pair.
        count_unit = "pairs" if self._match_mode in _PAIR_MODES else "claims"
        service = CheckingService(
            model=self._checker_model,
            extractor_model=_INTERNAL_EXT_MODEL,
            base_url=self._checker_base_url,
            concurrency=self._concurrency,
            prompt_cache=self._prompt_cache,
            templates=self._templates,
            joint=True,
            joint_num=self._bundle_size,
            max_words=self._max_words,
            verbosity="silent" if self._runs > 1 else "compact",
            section_label="Matching pass 1: GT → Pred (recall)",
            joint_prompt_key="claim_match_prompt",
            count_unit=count_unit,
        )

        # ── Pass 1: GT → Pred (recall / FN detection) ──────────
        pass1_items = self._build_pass_items(
            valid_items, claims_key=self._gt_key, ref_key=self._pred_key
        )

        await service.run(pass1_items)

        # ── Pass 2: Pred → GT (precision / FP detection) ───────
        pass2_items = self._build_pass_items(
            valid_items, claims_key=self._pred_key, ref_key=self._gt_key
        )

        # Fresh service to reset per-run stats
        service2 = CheckingService(
            model=self._checker_model,
            extractor_model=_INTERNAL_EXT_MODEL,
            base_url=self._checker_base_url,
            concurrency=self._concurrency,
            prompt_cache=self._prompt_cache,
            templates=self._templates,
            joint=True,
            joint_num=self._bundle_size,
            max_words=self._max_words,
            verbosity="silent" if self._runs > 1 else "compact",
            section_label="Matching pass 2: Pred → GT (precision)",
            joint_prompt_key="claim_match_prompt",
            count_unit=count_unit,
        )

        await service2.run(pass2_items)

        # ── Collect results ────────────────────────────────────
        verdict_key = f"{self._checker_model}_checker_verdict"
        explanation_key = f"{self._checker_model}_checker_explanation"

        cells1 = self._pass_cells(pass1_items, valid_items, self._gt_key, "prediction")
        cells2 = self._pass_cells(pass2_items, valid_items, self._pred_key,
                                  "ground-truth claim")

        results = []
        for i, item in enumerate(valid_items):
            gt_triplets = item[self._gt_key]
            pred_triplets = item[self._pred_key]

            # The counting reads a verdict and an explanation per claim:
            # the service's own in many-to-many, the folded one in joint/strict.
            def checked(cells: list[dict]) -> list[dict]:
                return [{verdict_key: c["verdict"], explanation_key: c["explanation"]}
                        for c in cells]

            # Pass 1 (recall): GT triplets the predictions failed to cover → FN
            tp_from_recall, fn_list, unjudged_gt = self._count_pass(
                checked(cells1[i]), gt_triplets,
                verdict_key, explanation_key, "gt_triplet",
            )

            # Pass 2 (precision): predictions the GT does not support → FP
            tp_from_precision, fp_list, unjudged_pred = self._count_pass(
                checked(cells2[i]), pred_triplets,
                verdict_key, explanation_key, "pred_triplet",
            )

            # RAGChecker: two independent coverage counts, no shared TP, no min().
            #   recall  side: GT triplets entailed by predictions (tp_from_recall)
            #   precision side: pred triplets entailed by GT      (tp_from_precision)
            # FN/FP derive from the judged misses alone — unjudged claims
            # (checker failure) leave numerator and denominator entirely.
            results.append(_ItemMatchResult(
                tp_recall=tp_from_recall,
                tp_precision=tp_from_precision,
                fp=len(fp_list), fn=len(fn_list),
                false_positives=fp_list,
                false_negatives=fn_list,
                unjudged_gt=unjudged_gt,
                unjudged_pred=unjudged_pred,
                pred2gt=cells1[i],
                gt2pred=cells2[i],
            ))

        return results

    def _pass_cells(
        self,
        pass_items: list[dict],
        valid_items: list[dict],
        claims_key: str,
        other_side: str,
    ) -> list[list[dict]]:
        """One verdict cell per claim, per item, parallel to the claims.

        many-to-many: the service's verdict as it is (``_flat_cell``), one synthetic
        item per answer. joint/strict: one synthetic item per (answer,
        reference claim); each claim's pairs are gathered in reference order
        and folded (``_fold_pairs``). *other_side* names the reference claims
        in a miss's explanation ("prediction", "ground-truth claim").
        """
        service_kg_key = f"{_INTERNAL_EXT_MODEL}_response_kg"
        namespace = f"{self._checker_model}_checker"
        if self._match_mode not in _PAIR_MODES:
            return [[_flat_cell(t, namespace) for t in synth[service_kg_key]]
                    for synth in pass_items]

        pairs: list[list[list[dict]]] = [
            [[] for _ in item[claims_key]] for item in valid_items]
        for synth in pass_items:
            i, k = synth["_item_index"], synth["_ref_index"]
            for j, t in enumerate(synth[service_kg_key]):
                pairs[i][j].append({"index": k, **_flat_cell(t, namespace)})
        return [[self._fold_pairs(claim_pairs, other_side) for claim_pairs in item_pairs]
                for item_pairs in pairs]

    @staticmethod
    def _fold_pairs(pairs: list[dict], other_side: str = "claim of the other side") -> dict:
        """Fold one claim's pair verdicts to the claim's verdict.

        One-to-one: the claim is found when one single reference claim makes
        it true. The fold is ragcheck's over a matrix row (``_row_entailed``),
        with Contradiction kept apart from Neutral:

            any pair Entailment                      → Entailment (no-verdict pairs do not matter)
            no Entailment, a pair without a verdict  → None: unjudged, never a miss
            no Entailment, all judged, a Contradiction → Contradiction
            no Entailment, all judged, none          → Neutral

        The cell keeps the flat shape {verdict, explanation, error?, retry?}
        that the counting and the findings read, and adds ``matched_by`` (the
        indexes of the reference claims that made it true) and ``pairs`` (one
        cell per reference claim, ``index`` = its position in the other
        side's list, 0-based; the matcher saw it as R{index + 1}).

        ``explanation`` is always a string: the deciding pair's for an
        Entailment, one sentence for a miss or a missing verdict; the pairs
        carry the detail. ``retry`` is the deciding pair's round, so only an
        Entailment can carry it; a miss is decided by every pair, and each
        pair keeps its own (as faithcheck's ungrounded findings do).
        """
        verdicts = [p["verdict"] for p in pairs]
        matched_by = [p["index"] for p in pairs if p["verdict"] == _ENTAILMENT]
        if matched_by:
            decider = next(p for p in pairs if p["verdict"] == _ENTAILMENT)
            cell = {"verdict": _ENTAILMENT,
                    "explanation": decider.get("explanation")
                    or f"R{decider['index'] + 1} makes the claim true."}
            if decider.get("retry"):
                cell["retry"] = decider["retry"]
        elif None in verdicts:
            failed = [p for p in pairs if p["verdict"] is None]
            if len(failed) == len(pairs):
                why = f"No verdict: no pair of the {len(pairs)} returned one."
            else:
                why = (f"No verdict: {len(failed)} of {len(pairs)} pairs returned "
                       f"none, and no other pair makes the claim true.")
            cell = {"verdict": None, "explanation": why,
                    "error": failed[0].get("error") or "checker_failure"}
        else:
            contradictions = verdicts.count(_CONTRADICTION)
            neutral = len(verdicts) - contradictions
            tally = ", ".join(part for part in (
                f"{neutral} Neutral" if neutral else "",
                f"{contradictions} Contradiction" if contradictions else "") if part)
            cell = {"verdict": _CONTRADICTION if contradictions else "Neutral",
                    "explanation": (f"No single {other_side} makes the claim "
                                    f"true ({tally}).")}
        cell["matched_by"] = matched_by
        cell["pairs"] = pairs
        return cell

    @staticmethod
    def _count_pass(
        checked: list[dict],
        originals: list[dict],
        verdict_key: str,
        explanation_key: str,
        triplet_field: str,
    ) -> tuple[int, list[dict], list[dict]]:
        """Count one matching pass: checker verdicts → (entailed, misses, unjudged).

        Three-way split:
            "Entailment"       → entailed (TP for this pass)
            other real verdict → miss (a judged disagreement: FN or FP)
            None               → unjudged (the checker never returned a verdict
                                 after all retries — a tooling failure)

        A claim nobody judged is no evidence about the extractor: unjudged
        claims leave numerator AND denominator, same rule as ragcheck's
        None-verdict propagation. They must never be charged as misses.

        Args:
            checked: Triplets carrying the pass's verdicts (parallel to originals).
            originals: The original triplets of this pass's claims side.
            triplet_field: Key name for the triplet string in miss/unjudged
                entries ("gt_triplet" for pass 1, "pred_triplet" for pass 2).
        """
        entailed = 0
        misses: list[dict] = []
        unjudged: list[dict] = []
        for j, t in enumerate(checked):
            verdict = t.get(verdict_key)
            if verdict == "Entailment":
                entailed += 1
            elif verdict is None:
                unjudged.append({
                    triplet_field: ExtractorEvaluator._triplet_to_str(originals[j]),
                    "cause": "checker_failure",
                })
            else:
                misses.append({
                    triplet_field: ExtractorEvaluator._triplet_to_str(originals[j]),
                    "verdict": verdict,
                    "reason": t.get(explanation_key) or verdict,
                })
        return entailed, misses, unjudged

    def _build_pass_items(
        self,
        valid_items: list[dict],
        claims_key: str,
        ref_key: str,
    ) -> list[dict]:
        """Build synthetic items for a checker pass.

        Triplets from claims_key become the claims to check, triplets from
        ref_key the reference text, each line labelled with its real number:
        the third reference claim is ``(R3)`` in every mode, so an
        explanation that names it can be read against the record.

        many-to-many: one item per answer, the whole reference list as its reference.
        joint/strict: one item per (answer, reference claim), that one line
        as its reference, over fresh copies of the claims (the same verdict
        key would collide across reference claims otherwise, as in
        ragcheck's matrix directions). ``_item_index`` / ``_ref_index`` say
        which pair an item is; the service ignores them.
        """
        service_kg_key = f"{_INTERNAL_EXT_MODEL}_response_kg"

        def claims(item: dict) -> list[dict]:
            # Fresh s/p/o dicts: no verdict of an earlier run rides along.
            return [{"subject": t["subject"], "predicate": t["predicate"],
                     "object": t["object"]} for t in item[claims_key]]

        def ref_line(k: int, t: dict) -> str:
            # R prefix, not [n]: claims carry [n], and a bare number on both
            # sides reads as a cross-reference the model then tries to honour.
            return f"(R{k+1}) {self._triplet_to_str(t)}"

        pass_items = []
        for i, item in enumerate(valid_items):
            ref_triplets = item[ref_key]
            if self._match_mode in _PAIR_MODES:
                for k, t in enumerate(ref_triplets):
                    pass_items.append({"reference": [ref_line(k, t)],
                                       service_kg_key: claims(item),
                                       "_item_index": i, "_ref_index": k})
                continue
            ref_text = "\n".join(ref_line(k, t) for k, t in enumerate(ref_triplets))
            pass_items.append({"reference": [ref_text], service_kg_key: claims(item)})

        return pass_items

    @staticmethod
    def _triplet_to_str(triplet: dict) -> str:
        """Convert a triplet dict to a natural-language string."""
        return f"{triplet['subject']} {triplet['predicate']} {triplet['object']}"

    @staticmethod
    def _fmt_ratio(
        value: float | None, numerator: int, denominator: int,
        judged: bool = False,
    ) -> str:
        """Format a metric with its fraction; None = empty denominator.

        *judged* marks a denominator smaller than the tree header total
        (unjudged claims left both sides of the ratio)."""
        if value is None:
            return f"n/a  ({numerator} / {denominator} — nothing judged)"
        suffix = " judged" if judged else ""
        return f"{value:.3f}  ({numerator} / {denominator}{suffix})"

    # ── Disagreement collection ──────────────────────────────────

    def _build_items(
        self,
        buckets: _ItemBucket,
        item_results: list[_ItemMatchResult],
    ) -> list[dict]:
        """The per-item record: every attempted item with its bucket and
        both claim lists as clean s/p/o, as ragcheck writes them. Compared
        items add the verdict cells, parallel to the claims and named like
        ragcheck's directions, reference first: ``pred2gt`` (pass 1, GT
        claims against the predictions) and ``gt2pred`` (pass 2). The other
        buckets never reached the matcher, so they have none."""
        def spo(triplets: list[dict]) -> list[dict]:
            return [_spo(t) for t in triplets]

        def entry(item: dict, i: int, bucket: str, **extra) -> dict:
            # Sparse: only an item extracted with its question says so.
            given = {"question_given": True} if self._gets_question(item) else {}
            return {
                "id": item.get("id", f"{bucket}-{i}"),
                "question": item.get("question", ""),
                "response": item.get("response", ""),
                **given,
                "bucket": bucket,
                **extra,
            }

        items: list[dict] = []
        for i, (item, ir) in enumerate(zip(buckets.to_compare, item_results)):
            items.append(entry(item, i, "compared",
                               gt_claims=spo(item[self._gt_key]),
                               pred_claims=spo(item[self._pred_key]),
                               pred2gt=ir.pred2gt, gt2pred=ir.gt2pred))
        for i, item in enumerate(buckets.answer_missed):
            items.append(entry(item, i, "answer_missed",
                               gt_claims=spo(item[self._gt_key]), pred_claims=[]))
        for i, item in enumerate(buckets.abstention_recognized):
            items.append(entry(item, i, "abstention_recognized",
                               gt_claims=[], pred_claims=[]))
        for i, item in enumerate(buckets.abstention_misread):
            items.append(entry(item, i, "abstention_misread",
                               gt_claims=[], pred_claims=spo(item[self._pred_key])))
        for i, item in enumerate(buckets.extraction_error):
            items.append(entry(item, i, "extraction_error",
                               gt_claims=spo(item.get(self._gt_key) or []),
                               pred_claims=[],
                               cause=item.get(self._error_key, "unknown")))
        return items

    @staticmethod
    def _build_findings(items: list[dict]) -> dict:
        """The review queue: the console branches opened up. Matching
        Quality — ``missed`` (GT claim the predictions do not cover),
        ``unsupported`` (prediction the GT does not support), ``unjudged``
        (no verdict; ``side`` names the pass). Abstention Handling —
        ``answer_missed`` (GT claims lost to an empty extraction),
        ``abstention_misread`` (claims invented from a refusal).
        Reliability — ``extraction_failed`` with its cause. A pure view
        over the record's items.

        A matching entry carries what it is judged against, as the checker
        eval's carry their chunks: the response and both claim lists. Claims
        stay s/p/o, unlike ragcheck's text: this queue is reviewed to repair
        the ground truth, and a missing claim goes in as the triplet it names.

        In joint and strict a cell holds its pairs; the entry then shows them
        (``pairs``): each reference claim with its verdict and explanation,
        so the reviewer sees why no single claim matched."""
        def pairs_of(cell: dict, references: list[dict]) -> dict:
            if "pairs" not in cell:
                return {}
            return {"pairs": [{"index": p["index"],
                               "reference_claim": references[p["index"]],
                               **{k: v for k, v in p.items() if k != "index"}}
                              for p in cell["pairs"]]}

        def classify(item: dict):
            head = {"id": item["id"], "question": item["question"]}
            bucket = item["bucket"]
            if bucket == "compared":
                against = {"response": item["response"],
                           "gt_claims": item["gt_claims"],
                           "pred_claims": item["pred_claims"]}
                for side, claims, cells, miss, references in (
                        ("gt", "gt_claims", "pred2gt", "missed", "pred_claims"),
                        ("pred", "pred_claims", "gt2pred", "unsupported", "gt_claims")):
                    for claim, cell in zip(item[claims], item[cells]):
                        if cell["verdict"] is None:
                            yield "unjudged", {**head, "claim": claim, "side": side,
                                               "cause": cell.get("error", "checker_failure"),
                                               **pairs_of(cell, item[references]),
                                               **against}
                        elif cell["verdict"] != "Entailment":
                            yield miss, {**head, "claim": claim,
                                         "verdict": cell["verdict"],
                                         "explanation": cell["explanation"], **retry_of(cell),
                                         **pairs_of(cell, item[references]),
                                         **against}
            elif bucket == "answer_missed":
                yield "answer_missed", {**head, "response": item["response"],
                                        "claims": item["gt_claims"]}
            elif bucket == "abstention_misread":
                yield "abstention_misread", {**head, "response": item["response"],
                                             "claims": item["pred_claims"]}
            elif bucket == "extraction_error":
                yield "extraction_failed", {**head, "cause": item["cause"]}

        return findings_view(
            ["missed", "unsupported", "answer_missed", "abstention_misread",
             "unjudged", "extraction_failed"], items, classify)

    # ── Logging (evaluator-owned sections) ───────────────────────

    def _log_data_pre(
        self, total: int, dropped: int, valid: int
    ) -> None:
        """Print 📂 Data section — pre-extraction validation summary."""
        logger.info(" 📂 Data")
        logger.info("    Total:        %d items", total)
        no_response = getattr(self, "_dropped_missing_response", dropped)
        no_gt = getattr(self, "_dropped_missing_gt", 0)
        skipped = getattr(self, "_skipped_marked", 0)
        if skipped > 0:
            logger.info("     ├─ skipped:  %d  (marked '%s')", skipped, self._skip_marked)
        if no_response > 0:
            logger.info("     ├─ dropped:  %d  (missing response)", no_response)
        if no_gt > 0:
            logger.info("     ├─ dropped:  %d  (missing GT key '%s')",
                        no_gt, self._gt_key)
        no_question = getattr(self, "_dropped_missing_question", 0)
        if no_question > 0:
            logger.info("     ├─ dropped:  %d  (marked '%s', no 'question')",
                        no_question, self._question_if)
        gt_empty = getattr(self, "_gt_empty", 0)
        logger.info("     └─ valid:    %d items  (%d with GT claims, %d %s)",
                    valid, valid - gt_empty, gt_empty,
                    plural(gt_empty, "abstention"))
        logger.info("")

    def _log_eval_config(self, items: list[dict] | None = None) -> None:
        """Print ⚙️  Config section — evaluator-specific. The Question line
        shows only when *items* get the question prompt (the service runs
        compact and prints no Config of its own)."""
        logger.info(" ⚙️  Config")

        ext_location = self._extractor_model
        if self._extraction_service.base_url:
            ext_location += f" @ {self._extraction_service.base_url}"
        logger.info("    Extractor:   %s", ext_location)
        logger.info("    GT key:      %s", self._gt_key)

        location = self._checker_model
        if self._checker_base_url:
            location += f" @ {self._checker_base_url}"
        logger.info("    Matching:    LLM 2-pass (%s)", location)
        logger.info("    Match mode:  %s", self._match_mode_line())

        if self._atomization_service is not None:
            logger.info("    Atomicity:   %s", self._atomizer_model)
        else:
            logger.info("    Atomicity:   skipped (%s)", self._atomizer_skip_reason)

        logger.info("    Prompts:     %s", prompts.describe_prompts(self._templates))
        items = items or []
        marked = sum(self._gets_question(item) for item in items)
        if marked:
            logger.info("    Question:    %d of %d items get the question prompt ('%s')",
                        marked, len(items), self._question_if)
        logger.info("")

    def _log_eval_results(self, result: ExtractorEvalResult) -> None:
        """Print ── EXTRACTOR EVAL ── section: P/R/F1, stats, abstentions."""
        if self._runs > 1:
            logger.info("")
        else:
            logger.info(settings.section_rule("EXTRACTOR EVAL", char="═"))
            logger.info("")

        # ── Matching quality: one funnel per side. Every issued claim lands
        # in exactly one branch; the first three sum to the denominator, the
        # 💥 branch is explicitly outside it.
        rc = result.recall_counts
        pc = result.precision_counts
        ah = result.abstention_handling

        logger.info(" 🔎 Matching Quality  (LLM 2-pass)")
        logger.info("    Recall — %d total GT claims", rc["total_gt_claims"])
        logger.info("     ├─ ✅ %d covered by predictions  (judged)", rc["covered"])
        logger.info("     ├─ ❌ %d missed  (judged)", rc["missed"])
        if rc["answer_missed_penalty"]:
            logger.info(
                "     ├─ ⚪ %d answer-missed penalty  (%d %s, nothing extracted for %d %s)",
                rc["answer_missed_penalty"],
                ah["answers_missed"], plural(ah["answers_missed"], "item"),
                rc["answer_missed_penalty"],
                plural(rc["answer_missed_penalty"], "claim"),
            )
        else:
            logger.info("     ├─ ⚪ 0 answer-missed penalty")
        if rc["unjudged"]:
            logger.info(
                "     ├─ 💥 %d unjudged by checker  (excluded from evaluation — no verdict returned)",
                rc["unjudged"],
            )
        else:
            logger.info("     ├─ 💥 0 unjudged by checker")
        logger.info("     └─ → Recall %s", self._fmt_ratio(
            result.recall, rc["covered"], rc["denominator"],
            judged=bool(rc["unjudged"])))

        logger.info("    Precision — %d total predicted claims", pc["total_pred_claims"])
        logger.info("     ├─ ✅ %d supported by GT  (judged)", pc["supported"])
        logger.info("     ├─ ❌ %d unsupported  (judged)", pc["unsupported"])
        if pc["abstention_misread_penalty"]:
            logger.info(
                "     ├─ ⚪ %d abstention-misread penalty  (%d %s, no GT for %d %s)",
                pc["abstention_misread_penalty"],
                ah["abstentions_misread"], plural(ah["abstentions_misread"], "item"),
                pc["abstention_misread_penalty"],
                plural(pc["abstention_misread_penalty"], "claim"),
            )
        else:
            logger.info("     ├─ ⚪ 0 abstention-misread penalty")
        if pc["unjudged"]:
            logger.info(
                "     ├─ 💥 %d unjudged by checker  (excluded from evaluation — no verdict returned)",
                pc["unjudged"],
            )
        else:
            logger.info("     ├─ 💥 0 unjudged by checker")
        logger.info("     └─ → Precision %s", self._fmt_ratio(
            result.precision, pc["supported"], pc["denominator"],
            judged=bool(pc["unjudged"])))

        logger.info("    f1: %s  (harmonic mean of recall · precision)",
                    "n/a" if result.f1 is None else f"{result.f1:.3f}")

        # ── Extraction stats
        logger.info("")
        logger.info(" 📊 Extraction Stats")
        logger.info(
            "    GT:    %d claims across %d items  (avg %.1f/item)",
            result.gt_stats["total_triplets"],
            result.to_compare_items,
            result.gt_stats["avg_per_item"],
        )
        logger.info(
            "    Pred:  %d claims across %d items  (avg %.1f/item)",
            result.pred_stats["total_triplets"],
            result.to_compare_items,
            result.pred_stats["avg_per_item"],
        )
        delta = result.pred_stats["avg_per_item"] - result.gt_stats["avg_per_item"]
        if abs(delta) > 0.05:
            direction = "over-extraction" if delta > 0 else "under-extraction"
            logger.info("    Delta: %+.1f/item  (%s)", delta, direction)

        # ── Abstention handling (own dimension: how the extractor treats
        # each response kind; tooling failures are excluded from this
        # universe by _classify — one number, one home).
        errored = (result.extraction_errors or {}).get("count", 0)
        note = "abstention = the response asserts nothing, GT has no claims"
        if errored:
            note += f"; {errored} extraction-failed excluded — see 💥 Reliability"
        logger.info("")
        log_mece_tree(
            "⚪ Abstention Handling", ah["answers"] + ah["abstentions"],
            "evaluated items",
            [
                ("💬", ah["answers"], plural(ah["answers"], "answer"), None,
                 [("extracted", ah["answers_extracted"],
                   "compared in Matching Quality"),
                  ("nothing extracted", ah["answers_missed"],
                   "every GT claim missed — charged in recall")]),
                ("🤐", ah["abstentions"], plural(ah["abstentions"], "abstention"), None,
                 [("recognized", ah["abstentions_recognized"],
                   "nothing extracted — correct"),
                  ("misread", ah["abstentions_misread"],
                   "claims invented from a refusal — charged in precision")]),
            ],
            footer=[("abstention recognized rate", ah["abstentions_recognized"],
                     ah["abstentions"]),
                    ("abstention misread rate", ah["abstentions_misread"],
                     ah["abstentions"]),
                    ("answer missed rate", ah["answers_missed"], ah["answers"])],
            header_note=note,
        )

        # ── Reliability — the eval's own tooling, co-equal with P/R/F1 but
        # never mixed into it. Rate keys are the report's JSON paths; they
        # unify under the canonical run-document later.
        ee = result.extraction_errors or {"count": 0, "rate": 0.0, "by_cause": {}}
        cf = result.checker_failures
        behavioral = ah["answers"] + ah["abstentions"]
        rows = [
            ("📝", "Extraction (subject)", ee["count"], behavioral + ee["count"],
             "items failed", "extraction_error_rate", ee.get("by_cause") or None),
            ("🔎", "Matching checker", cf["count"], cf["issued_verdicts"],
             "verdicts unjudged", "checker_failure_rate", None,
             (f"measurement is partial — run `eval checker` to qualify"
              f" '{self._checker_model}'") if cf["count"] > 0 else None),
        ]
        a_dim = result.atomicity
        if a_dim:
            rows.append(("🧬", "Atomization", a_dim.get("failed", 0),
                         a_dim["extracted_claims"], "claims failed",
                         "atomization_failure_rate", None))
        else:
            rows.append(("🧬", "Atomization", None, 0,
                         self._atomizer_skip_reason or "not configured",
                         None, None))
        logger.info("")
        log_rate_rows(
            "💥 Reliability", rows,
            header_note="tooling — excluded from all metrics, counted once here",
        )

        # ── Atomicity (orthogonal to coverage; only if measured)
        a = result.atomicity
        if a:
            logger.info("")
            logger.info(" 🧬 Atomicity")
            if a.get("failed"):
                logger.info("    ⚠️  %d claims failed atomization (omitted from stats)", a["failed"])
            logger.info(
                "    Evaluated:   %d claims → %d atomic units",
                a.get("evaluated_claims", a["extracted_claims"] - a.get("failed", 0)), 
                a["atomic_units"],
            )
            logger.info(
                "    Non-atomic:  %d  → atomicity_rate %.3f",
                a["non_atomic"], a["atomicity_rate"],
            )
            logger.info("    Claim density:  %.2f facts/claim  → claim_density %.2f",
                        a["information_density"], a["information_density"])

        # ── Duplicates (orthogonal to coverage; read-only, never deduped here)
        d = result.duplicates
        if d:
            logger.info("")
            logger.info(" 🔁 Duplicates")
            logger.info(
                "    Found %d exact duplicates of %d predicted"
                "  → duplicate_rate %.3f",
                d["duplicate_claims"], d["predicted_claims"], d["duplicate_rate"],
            )
            for entry in d["items"]:
                logger.info("    item %s:", entry["id"])
                for triplet_str in entry["duplicates"]:
                    logger.info("      • %s", triplet_str)

    def _log_done(self, run_doc: dict) -> None:
        """Print ✅ Done summary line — compared items and the headline metrics."""
        n = run_doc["counts"]["abstention_handling"]["answers_extracted"]
        logger.info("")
        logger.info(" ✅ Done: %d %s · %s", n, plural(n, "item"),
                    format_headline(run_doc["metrics"], self._RUN_SUMMARY_KEYS))
