"""
The retry marker: a verdict that came out of a retry round says which one.

Worker (ClaimVerdict.retry) → service ({namespace}_retry on the triplet) →
matrix fold ({namespace}_retries) → report cells ("retry"), plus
_meta.retry_rounds so the number stays readable. The report-builder tests
live next to each pipeline's own tests.
"""

import asyncio
from unittest.mock import patch

import pytest

from claimlens.models import CheckingPayload, DEFAULT_RETRY_ROUNDS, Direction
from claimlens.pipelines.directions import run_direction
from claimlens.services.atomization import AtomizationService
from claimlens.services.checking import CheckingService
from claimlens.services.extraction import ExtractionService
from claimlens.workers.checker import Checker, ClaimVerdict, Verdict

GARBAGE = "not json"


def _single(verdict, explanation="e"):
    return f'{{"explanation": "{explanation}", "verdict": "{verdict}"}}'


def _joint(*pairs):
    items = ", ".join(
        f'{{"claim_id": {cid}, "explanation": "e", "verdict": "{v}"}}' for cid, v in pairs)
    return f'{{"verdicts": [{items}]}}'


class ScriptedClient:
    """One scripted response list per generate_batch call, in order."""

    def __init__(self, *batches):
        self._batches = list(batches)
        self.last_batch_requests = 0

    async def generate_batch(self, tasks, description=None, task=None):
        batch = self._batches.pop(0)
        assert len(batch) == len(tasks)
        self.last_batch_requests = len(tasks)
        return batch


def _checker(*batches):
    w = Checker(api_key="k", model="m", base_url="http://fake/v1")
    w.client = ScriptedClient(*batches)
    return w


def _payloads(n):
    return [CheckingPayload(claim=f"c{i}", reference=["r"], item_index=0, claim_index=i)
            for i in range(n)]


# ── Worker ───────────────────────────────────────────────────────────────────

class TestWorkerSingleMode:

    def test_round_is_recorded_per_verdict(self):
        w = _checker(
            [_single("Entailment"), GARBAGE, GARBAGE],
            [_single("Neutral"), GARBAGE],
            [_single("Contradiction")],
        )
        out = asyncio.run(w.check_batch(_payloads(3)))
        assert [(v.verdict, v.retry) for v in out] == [
            (Verdict.ENTAILMENT, None), (Verdict.NEUTRAL, 1), (Verdict.CONTRADICTION, 2)]

    def test_permanent_failure_has_error_not_retry(self):
        w = _checker([GARBAGE], [GARBAGE], [GARBAGE])
        [v] = asyncio.run(w.check_batch(_payloads(1)))
        assert (v.verdict, v.error, v.retry) == (None, "parse_failure", None)


class TestWorkerJointMode:

    def test_only_the_recovered_gap_carries_the_round(self):
        chunk = ([(1, "c1"), (2, "c2")], ["r"])
        w = _checker(
            [_joint((1, "Entailment"))],        # claim 2 missing: an id gap
            [_joint((2, "Neutral"))],
        )
        [result] = asyncio.run(w.check_joint_batch([chunk]))
        assert (result[1].verdict, result[1].retry) == (Verdict.ENTAILMENT, None)
        assert (result[2].verdict, result[2].retry) == (Verdict.NEUTRAL, 1)

    def test_second_round_in_joint_mode(self):
        chunk = ([(1, "c1")], ["r"])
        w = _checker([GARBAGE], [GARBAGE], [_joint((1, "Contradiction"))])
        [result] = asyncio.run(w.check_joint_batch([chunk]))
        assert (result[1].verdict, result[1].retry) == (Verdict.CONTRADICTION, 2)


# ── Service ──────────────────────────────────────────────────────────────────

KG_KEY = "ext_response_kg"
RETRY_KEY = "chk_checker_retry"


@pytest.fixture
def service():
    with patch("claimlens.services.checking.Checker"):
        return CheckingService(model="chk", extractor_model="ext")


class TestServiceSerialize:

    def test_retry_key_is_part_of_the_output_contract(self, service):
        assert service.checker_retry_key == RETRY_KEY

    def test_retried_verdict_gets_the_key_first_pass_does_not(self, service):
        items = [{KG_KEY: [{"subject": "a", "predicate": "b", "object": "c"},
                           {"subject": "d", "predicate": "e", "object": "f"}]}]
        service._serialize(items, {0: {
            0: ClaimVerdict(verdict=Verdict.ENTAILMENT),
            1: ClaimVerdict(verdict=Verdict.NEUTRAL, retry=2),
        }})
        first, retried = items[0][KG_KEY]
        assert RETRY_KEY not in first
        assert retried[RETRY_KEY] == 2

    def test_rejudging_on_the_first_pass_clears_a_stale_marker(self, service):
        """A marker left over from an earlier judgement would claim a retry
        that never happened for the verdict now on the triplet."""
        triplet = {"subject": "a", "predicate": "b", "object": "c", RETRY_KEY: 2}
        items = [{KG_KEY: [triplet]}]
        service._serialize(items, {0: {0: ClaimVerdict(verdict=Verdict.ENTAILMENT)}})
        assert RETRY_KEY not in triplet


# ── Matrix fold ──────────────────────────────────────────────────────────────

NS = "chk_retrieved2response"


class RetryingService:
    """Stamps a verdict per shadow item; the second chunk's came from round 2."""

    kg_key = KG_KEY
    verdict_key = f"{NS}_verdict"
    explanation_key = f"{NS}_explanation"
    checker_error_key = f"{NS}_error"
    checker_retry_key = f"{NS}_retry"
    extraction_error_key = "ext_extraction_error"

    def __init__(self, retried_chunks):
        self._retried = retried_chunks

    async def run(self, data):
        for idx, item in enumerate(data):
            for t in item[KG_KEY]:
                t[self.verdict_key] = "Neutral"
                if idx in self._retried:
                    t[self.checker_retry_key] = 2
        return data


class TestMatrixFold:

    def _run(self, triplet, retried_chunks):
        items = [{KG_KEY: [triplet], "response": "r", "retrieved_context": ["c0", "c1"]}]
        direction = Direction(name="retrieved2response", kg_key=KG_KEY, per_chunk=True)
        asyncio.run(run_direction(RetryingService(retried_chunks), items, direction))
        return triplet

    def test_retries_are_sparse_per_chunk(self):
        t = self._run({"subject": "a", "predicate": "b", "object": "c"}, {1})
        assert t[f"{NS}_retries"] == {1: 2}

    def test_no_retry_leaves_no_map(self):
        t = self._run({"subject": "a", "predicate": "b", "object": "c"}, set())
        assert f"{NS}_retries" not in t

    def test_rejudged_chunk_drops_its_stale_entry(self):
        t = self._run({"subject": "a", "predicate": "b", "object": "c",
                       f"{NS}_retries": {0: 1, 1: 2}}, {1})
        assert t[f"{NS}_retries"] == {1: 2}


# ── _meta.retry_rounds and the guard that keeps it true ─────────────────────

class TestMetaRetryRounds:

    def test_describes_the_default_rounds(self):
        from claimlens.utils import describe_retry_rounds
        assert describe_retry_rounds(DEFAULT_RETRY_ROUNDS) == [
            {"prompt": "standard", "temperature": 0.3},
            {"prompt": "plain", "temperature": 0.5},
        ]

    @pytest.mark.parametrize("module, worker, build", [
        ("claimlens.services.checking", "Checker",
         lambda: CheckingService(model="m", extractor_model="e")),
        ("claimlens.services.extraction", "Extractor",
         lambda: ExtractionService(model="m")),
        ("claimlens.services.atomization", "Atomizer",
         lambda: AtomizationService(model="m", source_kg_key="k")),
    ])
    def test_no_service_passes_its_own_rounds(self, module, worker, build):
        """_meta.retry_rounds reports DEFAULT_RETRY_ROUNDS. That is only true
        while every worker in a report runs on the default — a service that
        passed its own rounds would make _meta lie, so it must fail here."""
        with patch(f"{module}.{worker}") as cls:
            build()
        assert "retry_rounds" not in cls.call_args.kwargs
        assert DEFAULT_RETRY_ROUNDS  # the constant _meta describes
