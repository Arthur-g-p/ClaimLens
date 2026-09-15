"""
claimlens — Claim-level evaluation for LLM outputs: decompose text into
atomic claims, then verify every claim against a reference.

The public API, stable within 1.x. The same five verbs as the CLI, with the
same keyword names as its flags:

    import claimlens
    record, findings = claimlens.ragcheck(items, extractor_model=..., checker_model=...)
    record, findings = claimlens.faithcheck(items, extractor_model=..., checker_model=...)
    record, findings = claimlens.refcheck(items, extractor_model=..., checker_model=...)
    items = claimlens.extract(items, extractor_model=...)
    items = claimlens.check(items, checker_model=..., extractor_model=...)

    entry = claimlens.check_faithfulness(response, chunks, extractor_model=..., checker_model=...)
    entry = await claimlens.acheck_faithfulness(response, chunks, extractor_model=..., checker_model=...)

    claimlens.enable_logging()      # output is silent until asked for
    claimlens.__version__

The batch verbs are synchronous. Inside a running event loop (a Jupyter
cell, an async server) they refuse and point here, the async door, which is
the same code:

    pipeline = RagCheckerPipeline(extractor_model=..., checker_model=...)
    await pipeline.run(items)
    record, findings = pipeline.last_report, pipeline.last_findings

Everything under claimlens.pipelines, .services and .workers is importable
but may change between minor versions.

The imports inside the functions are deliberate: ``import claimlens`` must
stay instant, and the pipeline stack pulls in litellm.
"""

import asyncio
from typing import NamedTuple

from claimlens.settings import enable_logging
from claimlens.utils import _package_version

__version__ = _package_version()


class Report(NamedTuple):
    """What a pipeline produces: the record (metrics, counts, every item's
    claims and verdicts) and the findings (the review queue derived from
    it). The same two documents the CLI writes to disk."""
    record: dict
    findings: dict


def _no_running_loop(verb: str, hint: str) -> None:
    """The sync verbs wrap asyncio.run, which cannot start inside a loop
    that is already running. Say so, and say what to use instead."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"claimlens.{verb}() was called inside a running event loop "
        f"(a Jupyter cell, an async server). Use the async form: {hint}"
    )


def _args(command: str, params: dict) -> dict:
    """The record's ``_args`` block, as the CLI writes it: what was asked
    for. From the library every parameter given is explicit."""
    return {"command": command, **params, "_explicit": sorted(params)}


def _report(pipeline, command: str, params: dict) -> Report:
    args = _args(command, params)
    return Report({"_args": args, **pipeline.last_report},
                  {"_args": args, **pipeline.last_findings})


# ── batch verbs: the CLI commands as functions ───────────────────────────

def ragcheck(items: list[dict], *, extractor_model: str, checker_model: str,
             **kwargs) -> Report:
    """RAGChecker-style RAG evaluation: two extractions, four checking
    directions, eleven metrics. Items need ``response``, ``gt_answer`` and
    ``retrieved_context``. Extra keyword arguments go to RagCheckerPipeline
    (``extractor_base_url``, ``checker_base_url``, ``concurrency``, ``joint``,
    ``runs``, ``verbosity``, ...). Returns (record, findings)."""
    _no_running_loop("ragcheck", "await RagCheckerPipeline(...).run(items)")
    from claimlens.pipelines.ragchecker import RagCheckerPipeline
    params = dict(extractor_model=extractor_model, checker_model=checker_model, **kwargs)
    pipeline = RagCheckerPipeline(**params)
    pipeline.run_sync(items)
    return _report(pipeline, "ragcheck", params)


def faithcheck(items: list[dict], *, extractor_model: str, checker_model: str,
               **kwargs) -> Report:
    """Faithfulness without ground truth: response claims against the
    retrieved context. Items need ``response`` and ``retrieved_context``.
    Returns (record, findings)."""
    _no_running_loop("faithcheck", "await FaithfulnessPipeline(...).run(items)")
    from claimlens.pipelines.faithfulness import FaithfulnessPipeline
    params = dict(extractor_model=extractor_model, checker_model=checker_model, **kwargs)
    pipeline = FaithfulnessPipeline(**params)
    pipeline.run_sync(items)
    return _report(pipeline, "faithcheck", params)


def refcheck(items: list[dict], *, extractor_model: str, checker_model: str,
             **kwargs) -> Report:
    """Reference checking: extraction plus checking in one pass. Items need
    ``response`` and ``reference``. Returns (record, findings)."""
    _no_running_loop("refcheck", "await RefCheckerPipeline(...).run(items)")
    from claimlens.pipelines.refchecker import RefCheckerPipeline
    params = dict(extractor_model=extractor_model, checker_model=checker_model, **kwargs)
    pipeline = RefCheckerPipeline(**params)
    pipeline.run_sync(items)
    return _report(pipeline, "refcheck", params)


def extract(items: list[dict], *, extractor_model: str,
            extractor_base_url: str | None = None, **kwargs) -> list[dict]:
    """Decompose each item's ``response`` into atomic claims, written back
    onto the items. Returns the enriched list, the input of ``check``."""
    _no_running_loop("extract", "await ExtractionService(...).run(items)")
    from claimlens.services.extraction import ExtractionService
    service = ExtractionService(model=extractor_model, base_url=extractor_base_url, **kwargs)
    return service.run_sync(items)


def check(items: list[dict], *, checker_model: str, extractor_model: str,
          checker_base_url: str | None = None, **kwargs) -> list[dict]:
    """Check already-extracted claims against each item's ``reference``.
    ``extractor_model`` names which extraction to read. Returns the
    enriched list."""
    _no_running_loop("check", "await CheckingService(...).run(items)")
    from claimlens.services.checking import CheckingService
    service = CheckingService(model=checker_model, extractor_model=extractor_model,
                              base_url=checker_base_url, **kwargs)
    return service.run_sync(items)


# ── one response, in real time ────────────────────────────────────────────

def check_faithfulness(*args, **kwargs) -> dict:
    """Score one response against its retrieved context, in process. Returns
    that item's report entry. See pipelines/faithfulness.py for the
    signature; ``acheck_faithfulness`` is the form for async callers."""
    _no_running_loop("check_faithfulness", "await claimlens.acheck_faithfulness(...)")
    from claimlens.pipelines.faithfulness import check_faithfulness as _impl
    return _impl(*args, **kwargs)


async def acheck_faithfulness(*args, **kwargs) -> dict:
    """The async twin of ``check_faithfulness``: same arguments, same return,
    for callers already inside an event loop."""
    from claimlens.pipelines.faithfulness import acheck_faithfulness as _impl
    return await _impl(*args, **kwargs)


__all__ = [
    "ragcheck", "faithcheck", "refcheck", "extract", "check",
    "check_faithfulness", "acheck_faithfulness",
    "enable_logging", "__version__", "Report",
]
