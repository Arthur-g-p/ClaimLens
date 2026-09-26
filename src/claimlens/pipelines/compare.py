"""
Compare - two texts, claim by claim, both ways: 2 extractions + 2 flat
checking directions. Each text's claims are checked against the other
text, never matched to the other text's claims: two extractions of the
same content take different forms (a summary weakens a claim instead of
dropping it), so claim-to-claim matching has no stable answer.

    a --extract--> {ext}_a_kg   ┐
    b --extract--> {ext}_b_kg   ┘ in place on the items

    b2a:  a claims vs the text of b    (flat)
    a2b:  b claims vs the text of a    (flat)

Direction names follow ragcheck's {reference}2{claims}. With a as the
ground truth these are ragcheck's two generator directions exactly:
a2b is answer2response (precision), b2a is response2answer (recall). The
numbers stay neutral (a_in_b, b_in_a, ...): which text is right is the
reader's call, made in the HTML report, so one run serves either reading.
"""

import time
from datetime import datetime

from claimlens import settings
from claimlens.exceptions import InvalidInputError
from claimlens.models import Direction
from claimlens.services.base import BaseService
from claimlens.services.extraction import ExtractionService
from claimlens.services.checking import CheckingService
from claimlens.pipelines.directions import (
    _CONTRADICTION,
    _ENTAILMENT,
    _flat_cell,
    _location,
    _ratio,
    _spo,
    log_pipeline_tree,
    pipeline_counts,
    run_direction,
    unwrap_items,
    verdict_summary,
)
from claimlens.stats import GLOBAL_STATS, format_headline, log_rate_rows, log_token_stats, usage_since
from claimlens.utils import build_meta, findings_view, plural

logger = settings.get_logger(__name__)

REQUIRED_KEYS = ("a", "b")
NAMES = {"a": "A", "b": "B"}

# Per-item metrics, A's side first: the share of a text's judged claims the
# other text states, and the share it contradicts.
METRIC_NAMES = ("a_in_b", "a_contradicted", "b_in_a", "b_contradicted")


def _missing_texts(item) -> list[str]:
    """Absent or null is missing; an empty string is data (a text that
    states nothing), as an empty response is everywhere else."""
    if not isinstance(item, dict):
        return list(REQUIRED_KEYS)
    return [k for k in REQUIRED_KEYS if not isinstance(item.get(k), str)]


# ── Metrics (pure functions - report entries in, numbers out) ────────────────

def compute_item_metrics(entry: dict) -> dict:
    """The four side metrics for one entry; None = not computable.

    An extraction error on either side excludes the whole item, the project
    rule for tooling failures. Unjudged claims leave numerator and
    denominator (unknown is not "no")."""
    metrics: dict = dict.fromkeys(METRIC_NAMES)
    if entry.get("extraction_errors"):
        return metrics
    for side, other, cells in (("a", "b", entry["b2a"]), ("b", "a", entry["a2b"])):
        known = [c["verdict"] for c in cells if c.get("verdict") is not None]
        metrics[f"{side}_in_{other}"] = _ratio(known.count(_ENTAILMENT), len(known))
        metrics[f"{side}_contradicted"] = _ratio(known.count(_CONTRADICTION), len(known))
    return metrics


def _verdict_cell_counts(results: list[dict]) -> tuple[int, int]:
    """(total, none) verdict cells across both directions, skipping
    extraction-errored items — shared by compute and display."""
    total = none = 0
    for e in results:
        if e.get("extraction_errors"):
            continue
        for key in ("b2a", "a2b"):
            for cell in e[key]:
                total += 1
                none += cell.get("verdict") is None
    return total, none


def compute_overall_metrics(results: list[dict]) -> dict:
    """Macro over items, skipping nulls, plus the two tooling rates —
    exactly the variance roster."""
    overall: dict = {}
    for name in METRIC_NAMES:
        values = [e["metrics"][name] for e in results
                  if e["metrics"].get(name) is not None]
        overall[name] = round(sum(values) / len(values), 4) if values else None
    evaluated = len(results)
    errored = sum(1 for e in results if e.get("extraction_errors"))
    total_cells, none_cells = _verdict_cell_counts(results)
    overall["extraction_error_rate"] = _ratio(errored, evaluated)
    overall["checker_failure_rate"] = _ratio(none_cells, total_cells)
    return overall


def compute_overall_counts(results: list[dict]) -> dict:
    """The numbers behind the console blocks — never varianced: ``support``
    (items behind each macro average) and ``reliability`` (the 💥 rows)."""
    support = {name: sum(1 for e in results if e["metrics"].get(name) is not None)
               for name in METRIC_NAMES}
    errored = [e for e in results if e.get("extraction_errors")]
    total_cells, none_cells = _verdict_cell_counts(results)
    return {
        "support": support,
        "reliability": {
            "extraction": {
                "failed": len(errored), "items": len(results),
                "by_cause": {
                    side: sum(1 for e in errored if e["extraction_errors"].get(side))
                    for side in REQUIRED_KEYS
                },
            },
            "checking": {"unjudged": none_cells, "issued": total_cells},
        },
    }


class ComparePipeline(BaseService):
    """2 extractions + 2 flat checking directions: two texts, both ways."""

    _RUN_SUMMARY_KEYS = ("a_in_b", "b_in_a")
    _VARIANCE_SECTIONS = {
        "metrics": [(NAMES["a"], ["a_in_b", "a_contradicted"]),
                    (NAMES["b"], ["b_in_a", "b_contradicted"])],
        "behavior": [],
        "health": ["extraction_error_rate", "checker_failure_rate"],
    }
    # Neutral words: without a ground truth a contradiction is a conflict
    # nobody is blamed for, and more overlap is neither better nor worse.
    _VARIANCE_LABELS = {"a_in_b": "A in B", "a_contradicted": "A conflicts with B",
                        "b_in_a": "B in A", "b_contradicted": "B conflicts with A"}
    _METRIC_DIRECTIONS = {"a_contradicted": "lower is better", "b_contradicted": "lower is better",
                          "extraction_error_rate": "lower is better",
                          "checker_failure_rate": "lower is better"}

    def __init__(
        self,
        extractor_model: str,
        checker_model: str,
        *,
        extractor_base_url: str | None = None,
        checker_base_url: str | None = None,
        concurrency: int = 10,
        dedup: bool = True,
        joint: bool = True,
        joint_num: int = settings.DEFAULT_JOINT_NUM,
        max_words: int | None = None,
        verbosity: str = "full",
        runs: int = 1,
    ):
        self._extractor_model = extractor_model
        self._checker_model = checker_model
        self._init_verbosity(verbosity)
        self._runs = max(1, runs)
        child_verbosity = (
            "silent" if (verbosity == "silent" or self._runs > 1) else "compact"
        )
        # The record and the findings, assembled by the base run loop from
        # the per-run entries _run_once leaves on last_run / last_run_findings.
        self.last_report: dict | None = None
        self.last_findings: dict | None = None
        self.last_run: dict | None = None
        self.last_run_findings: dict | None = None

        self._kg = {side: f"{extractor_model}_{side}_kg" for side in REQUIRED_KEYS}
        self._err = {side: f"{extractor_model}_{side}_extraction_error" for side in REQUIRED_KEYS}

        # Compose the services. Each fail-fasts on its own API key here.
        # mark_abstention=False on both: neither text is a response, so an
        # empty extraction says the text states nothing, not that it refused.
        extractor_config = dict(
            model=extractor_model,
            base_url=extractor_base_url,
            concurrency=concurrency,
            verbosity=child_verbosity,
            dedup=dedup,
            mark_abstention=False,
        )
        self._extract = {
            side: ExtractionService(
                **extractor_config,
                section_label=f"Extraction: {side}",
                source_key=side,
                kg_key=self._kg[side],
                error_key=self._err[side],
            )
            for side in REQUIRED_KEYS
        }

        self._directions: list[tuple[Direction, CheckingService]] = []
        for reference, claims in (("b", "a"), ("a", "b")):
            name = f"{reference}2{claims}"
            service = CheckingService(
                model=checker_model,
                extractor_model=extractor_model,
                base_url=checker_base_url,
                concurrency=concurrency,
                joint=joint,
                joint_num=joint_num,
                max_words=max_words,
                verbosity=child_verbosity,
                section_label=f"Direction: {name}",
                kg_key=self._kg[claims],
                verdict_namespace=f"{checker_model}_{name}",
                extraction_error_key=self._err[claims],
            )
            self._directions.append((
                Direction(name=name, kg_key=self._kg[claims], reference_key=reference),
                service,
            ))

    # -- Pipeline: the BaseService 7-step run() shape --

    async def run(self, data: list[dict]) -> list[dict]:
        """Run the pipeline N times (N = --runs, default 1); the base loop
        assembles last_report and last_findings."""
        return await self._run_repeated(data)

    async def _run_once(
        self, data: list[dict], announce: bool = True, report: bool = True,
    ) -> list[dict]:
        """One full pass over *data*, in place.

        1. Validate     - hard drop: a and b both strings
        2. Filter       - none (no skipping)
        3. Log pre-exec - validation + config
        4. Execute      - 2 extractions, then the 2 directions
        5. Serialize    - none in place; the run entry goes to last_run
        6. Log results  - consolidated results block
        7. Return mutated data
        """
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._started_perf = time.perf_counter()
        self._usage_at_start = GLOBAL_STATS.snapshot()
        data = unwrap_items(data)
        self._canonicalize_keys(data)
        valid = self._validate(data)
        self._filter(valid)
        if announce:
            self._log_validation(len(data), len(valid))
            self._log_config()

        # Children print their own labeled section rules (compact mode).
        for side in REQUIRED_KEYS:
            await self._extract[side].run(valid)

        self._prefill_known_verdicts(valid)
        for direction, service in self._directions:
            await run_direction(service, valid, direction)

        self._serialize()
        self.last_run = self._build_run(data)
        self.last_run_findings = {
            "_meta": dict(self.last_run["_meta"]),
            "findings": self._build_findings(self.last_run["items"]),
        }
        if report:
            self._log_results()
        return data

    # _run_repeated inherited from BaseService (variance mode)

    def _prefill_known_verdicts(self, items: list[dict]) -> None:
        """An empty text states nothing, so every claim of the other text is
        Neutral against it — written without a request, which the checking
        service then skips as already judged (as ragcheck does for an empty
        response or GT)."""
        for direction, service in self._directions:
            for item in items:
                if item[direction.reference_key].strip():
                    continue
                for triplet in item.get(direction.kg_key) or []:
                    triplet[service.verdict_key] = "Neutral"
                    triplet[service.explanation_key] = (
                        "the text is empty — not sent to the checker")

    # -- Validation --

    def _validate(self, data: list[dict]) -> list[dict]:
        """Step 1: Hard drop - both texts must be present as strings."""
        valid = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.debug("Item %d is not an object (%s) - skipping.",
                             i, type(item).__name__)
                continue
            missing = _missing_texts(item)
            if missing:
                logger.debug("Item %d missing %s - skipping.", i, ", ".join(missing))
                continue
            valid.append(item)

        if not valid:
            raise InvalidInputError("No items contain the two texts 'a' and 'b' (strings).")
        return valid

    def _filter(self, valid):
        """No skipping — a failed run is re-run from the start."""
        pass

    # -- Report (the single output artifact) --

    def _build_run(self, data: list[dict]) -> dict:
        """One ``runs[]`` entry of the record: {_meta, metrics, counts, items}.

        ``metrics`` are the varianced scalars (the console's Metrics roster),
        ``counts`` every number the console blocks print, ``items`` the
        complete per-item record. Pure projection of the mutated items —
        loss-free, no LLM calls, safe to rebuild anytime.
        """
        items = []
        dropped = 0
        for i, item in enumerate(data):
            if _missing_texts(item):
                dropped += 1
                continue
            items.append(self._build_result_entry(item, i))

        timestamp, duration = self._run_timing()
        meta = build_meta(
            "compare",
            timestamp=timestamp,
            duration_seconds=duration,
            total_items=len(data),
            evaluated_items=len(items),
            dropped_items=dropped,
            request_strategies=GLOBAL_STATS.strategies(),
            usage=usage_since(getattr(self, "_usage_at_start", None)),
        )
        counts = compute_overall_counts(items)
        counts = {"support": counts["support"],
                  "pipeline": pipeline_counts(self._pipeline_phases(items)),
                  "reliability": counts["reliability"]}
        return {"_meta": meta, "metrics": compute_overall_metrics(items),
                "counts": counts, "items": items}

    def _build_result_entry(self, item: dict, index: int) -> dict:
        """Claims lists first, then one verdict array per direction,
        parallel to the claims it judges (b2a to a_claims, a2b to b_claims)."""
        claims = {side: item.get(self._kg[side]) or [] for side in REQUIRED_KEYS}
        entry = {
            "id": str(item.get("id", f"item-{index}")),
            "a": item["a"],
            "b": item["b"],
            "a_claims": [_spo(t) for t in claims["a"]],
            "b_claims": [_spo(t) for t in claims["b"]],
        }
        for direction, _ in self._directions:
            namespace = f"{self._checker_model}_{direction.name}"
            entry[direction.name] = [_flat_cell(t, namespace)
                                     for t in item.get(direction.kg_key) or []]

        # Tooling failures surface explicitly - never mistakable for a text
        # that states nothing.
        errors = {side: item[self._err[side]] for side in REQUIRED_KEYS
                  if self._err[side] in item}
        if errors:
            entry["extraction_errors"] = errors

        entry["metrics"] = compute_item_metrics(entry)
        return entry

    def _pipeline_phases(self, items: list[dict]) -> list[tuple]:
        """The four phases, for the 🔀 tree and ``counts.pipeline`` alike:
        (icon, name, PhaseStats, summary text, tallies)."""
        phases: list[tuple] = []
        for side in REQUIRED_KEYS:
            n = sum(len(e[f"{side}_claims"]) for e in items)
            phases.append(("📝", f"extract {side}", self._extract[side].last_stats,
                           f"{n} {plural(n, 'claim')}", {"claims": n}))
        for direction, service in self._directions:
            c = {"total": 0, "Entailment": 0, "Contradiction": 0,
                 "Neutral": 0, "unknown": 0}
            for e in items:
                for cell in e[direction.name]:
                    c["total"] += 1
                    verdict = cell.get("verdict")
                    c[verdict if verdict in c else "unknown"] += 1
            tally = {"verdicts": c["total"], "Entailment": c["Entailment"],
                     "Contradiction": c["Contradiction"], "Neutral": c["Neutral"],
                     "unjudged": c["unknown"]}
            phases.append(("🔎", direction.name, service.last_stats,
                           verdict_summary(c), tally))
        return phases

    @staticmethod
    def _build_findings(items: list[dict]) -> dict:
        """The review queue, one list per claim outcome behind the metrics:
        ``a_not_in_b`` / ``b_not_in_a`` (the other text does not state the
        claim), ``a_contradicted`` / ``b_contradicted`` (the other text
        contradicts it), ``unjudged`` (no verdict, with its side and cause),
        ``extraction_failed``. Keys are neutral: which side is right is the
        reader's call. A pure view over the record's items."""
        def classify(item: dict):
            head = {"id": item["id"]}
            if item.get("extraction_errors"):
                for side, cause in item["extraction_errors"].items():
                    yield "extraction_failed", {**head, "side": side, "cause": cause}
                return
            for side, other, cells in (("a", "b", item["b2a"]), ("b", "a", item["a2b"])):
                for claim, cell in zip(item[f"{side}_claims"], cells):
                    entry = {**head, "claim": f"{claim['subject']} {claim['predicate']} {claim['object']}"}
                    verdict = cell.get("verdict")
                    if verdict is None:
                        yield "unjudged", {**entry, "side": side,
                                           "cause": cell.get("error", "checker_failure")}
                    elif verdict == _CONTRADICTION:
                        yield f"{side}_contradicted", {**entry, "explanation": cell.get("explanation")}
                    elif verdict != _ENTAILMENT:
                        yield f"{side}_not_in_{other}", {**entry, "explanation": cell.get("explanation")}

        return findings_view(
            ["a_not_in_b", "a_contradicted", "b_not_in_a", "b_contradicted",
             "unjudged", "extraction_failed"], items, classify)

    def _run_timing(self) -> tuple[str, float]:
        """(timestamp, elapsed) for the report envelope; safe before a run."""
        if not hasattr(self, "_started_at"):
            return datetime.now().isoformat(timespec="seconds"), 0.0
        return self._started_at, time.perf_counter() - self._started_perf

    # -- Serialization: none in place; last_run is the artifact --

    def _serialize(self, *args, **kwargs) -> None:
        pass

    # -- Logging --

    def _log_validation(self, total: int, valid: int) -> None:
        if self.verbosity != "full":
            return
        dropped = total - valid
        logger.info(" 📂 Validation")
        logger.info("    Total:        %d items", total)
        if dropped:
            logger.info("     ├─ dropped:  %d  (missing %s)",
                        dropped, "/".join(REQUIRED_KEYS))
        logger.info("     └─ valid:    %d items", valid)
        logger.info("")

    def _log_skip(self, *args, **kwargs) -> None:
        pass

    def _log_config(self) -> None:
        if self.verbosity != "full":
            return
        checking = self._directions[0][1]
        logger.info(" ⚙️  Config")
        logger.info("    Extractor:   %s", _location(self._extract["a"]))
        logger.info("    Checker:     %s", _location(checking))
        logger.info("    Mode:        %s", checking.mode_label)
        logger.info("    Directions:  %s",
                    ", ".join(d.name for d, _ in self._directions))
        logger.info("    Prompts:     %s", settings.PROMPT_PATH)
        logger.info("")

    def _log_results(self) -> None:
        self._log_bl_results()
        self._log_done()
        if self.verbosity == "full":
            log_token_stats()

    def _log_bl_results(self) -> None:
        """Print ── COMPARE RESULTS ──: pipeline tree + the metrics."""
        self._log_pipeline_tree()
        self._log_metrics()
        self._log_reliability()

    def _log_run_findings(self) -> None:
        """Per-run findings in variance mode: metrics + reliability — the
        pipeline tree (request plumbing) prints at --runs 1 but not per run."""
        logger.info("")
        self._log_metrics()
        self._log_reliability()

    def _log_pipeline_tree(self) -> None:
        """══ RESULTS ══ rule + 🔀 Pipeline: where the requests went."""
        if self.verbosity != "full":
            return
        logger.info(settings.section_rule("COMPARE RESULTS", char="═"))
        logger.info("")
        log_pipeline_tree(self._pipeline_phases(self.last_run["items"]))

    def _log_metrics(self) -> None:
        """📊 Metrics: one group per text, its claims checked against the
        other text's."""
        if self.verbosity != "full":
            return
        run = self.last_run
        om = run["metrics"]
        n = run["_meta"]["evaluated_items"]
        support = run["counts"]["support"]

        def fmt(name: str) -> str:
            value = om.get(name)
            text = "n/a" if value is None else f"{value:.3f}"
            notes = []
            if support.get(name) is not None and support[name] != n:
                notes.append(f"{support[name]} of {n} items")
            if value is not None and self._METRIC_DIRECTIONS.get(name):
                notes.append(self._METRIC_DIRECTIONS[name])
            if notes:
                text += f"  ({' · '.join(notes)})"
            return text

        labels = self._VARIANCE_LABELS
        width = max(len(label) for label in labels.values()) + 2
        logger.info(" 📊 Metrics  (macro over %d items)", n)
        for side, other in (("a", "b"), ("b", "a")):
            logger.info("    %s — its claims checked against the text of %s",
                        NAMES[side], NAMES[other])
            for metric, last in ((f"{side}_in_{other}", False), (f"{side}_contradicted", True)):
                logger.info("     %s %-*s%s", "└─" if last else "├─", width,
                            f"{labels[metric]}:", fmt(metric))
        logger.info("")

    def _log_reliability(self) -> None:
        """💥 Reliability rate rows — harness health, always printed."""
        if self.verbosity != "full":
            return
        rel = self.last_run["counts"]["reliability"]
        ext, chk = rel["extraction"], rel["checking"]
        causes = {NAMES[side]: n for side, n in ext["by_cause"].items() if n} or None
        log_rate_rows(
            "💥 Reliability",
            [("📝", "Extraction", ext["failed"], ext["items"],
              "items failed", "extraction_error_rate", causes),
             ("🔎", "Checking", chk["unjudged"], chk["issued"],
              "verdicts unjudged", "checker_failure_rate", None)],
            header_note="tooling — excluded from all metrics,"
                        " counted once here",
        )
        logger.info("")

    def _log_done(self) -> None:
        if self.verbosity != "full":
            return
        run = self.last_run
        n = run["_meta"]["evaluated_items"]
        logger.info(" ✅ Done: %d %s · %s", n, plural(n, "item"),
                    format_headline(run["metrics"], self._RUN_SUMMARY_KEYS))
