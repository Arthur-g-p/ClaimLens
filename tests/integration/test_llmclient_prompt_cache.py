"""
Integration tests for the prompt-cache marker and the live cache lines.

Workers send the user message as two text blocks: the prompt's fixed
instructions, then the item's input. The client marks the instructions block
cacheable on both paths; with no marker (prompt_cache=False, or rejected) it
joins the blocks into the one string the prompt always was. The marker must
never break a run: a 400 on a marked request while the endpoint is unknown
re-sends the SAME request as that joined string. Success then means the endpoint rejects the
marker (learned process-wide, never sent again); a second failure means the
marker was not the cause, and the error is handled as before.

Reading is the standard field usage.prompt_tokens_details.cached_tokens only.
Live lines are events only: the first hit; no hit in the first
CACHE_CHECK_AFTER answers; CACHE_CHECK_AFTER misses in a row after hitting;
and a first answer that does not report cache use at all.

All tests mock the SDKs — zero real network.

Run with:
    pytest tests/integration/test_llmclient_prompt_cache.py -v
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai import BadRequestError
from pydantic import BaseModel

from claimlens.llmclient import LLMClient, CACHE_CHECK_AFTER, RETRY_MATRIX
from claimlens.stats import GLOBAL_STATS


BASE_URL = "http://fake/v1"
MODEL = "test-model"
MARKER = {"type": "ephemeral"}
ROLE = "Role."
INSTRUCTIONS = "Fixed instructions. "
ITEM = "item text"


def _two_blocks(instructions: str, item: str) -> list[dict]:
    return [{"role": "system", "content": ROLE},
            {"role": "user", "content": [{"type": "text", "text": instructions},
                                         {"type": "text", "text": item}]}]


MESSAGES = _two_blocks(INSTRUCTIONS, ITEM)
JOINED = [{"role": "system", "content": ROLE}, {"role": "user", "content": INSTRUCTIONS + ITEM}]


# ── Helpers ──────────────────────────────────────────────────────


def _response(cached=None, prompt_tokens=1000) -> MagicMock:
    """A completion whose usage reports `cached` tokens; None = field absent."""
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 1}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = '{"ok": true}'
    r.usage.model_dump.return_value = usage
    return r


def _bad_request(message: str) -> BadRequestError:
    resp = MagicMock()
    resp.status_code = 400
    resp.headers = {}
    return BadRequestError(message=message, response=resp, body={"error": {"message": message}})


def _make_client(*, discovered: bool = True, prompt_cache: bool = True) -> LLMClient:
    with patch("claimlens.llmclient.AsyncOpenAI"):
        c = LLMClient(api_key="k", model=MODEL, base_url=BASE_URL, prompt_cache=prompt_cache)
    c._connection_verified = True
    if discovered:
        c._strategy_discovered = True
        c._discovery_succeeded = True
        c._strategy_index = 1  # 'Schema Only' — no reasoning, so no drop_params
    c.client.chat.completions.create = AsyncMock()
    return c


def _sent(call) -> list[dict]:
    return call.kwargs["messages"]


def _is_marked(messages: list[dict]) -> bool:
    content = messages[-1]["content"]
    return isinstance(content, list) and content[-2].get("cache_control") == MARKER


@pytest.fixture(autouse=True)
def _reset_process_state():
    for cache in (LLMClient._STRATEGY_CACHE, LLMClient._DROP_PARAMS_CACHE, LLMClient._VERIFIED_ENDPOINTS):
        cache.clear()
    yield
    for cache in (LLMClient._STRATEGY_CACHE, LLMClient._DROP_PARAMS_CACHE, LLMClient._VERIFIED_ENDPOINTS):
        cache.clear()


# ── What is sent ─────────────────────────────────────────────────


@pytest.mark.integration
class TestMarker:

    async def test_instructions_block_is_marked_item_block_is_not(self):
        c = _make_client()
        c.client.chat.completions.create.return_value = _response()

        await c.generate(MESSAGES)

        sent = _sent(c.client.chat.completions.create.call_args)
        assert sent[0] == {"role": "system", "content": ROLE}          # role line untouched
        assert sent[1]["content"] == [
            {"type": "text", "text": INSTRUCTIONS, "cache_control": MARKER},
            {"type": "text", "text": ITEM}]
        assert "cache_control" not in MESSAGES[1]["content"][0]         # caller's list not mutated

    async def test_messages_without_blocks_go_out_unchanged(self):
        c = _make_client()
        c.client.chat.completions.create.return_value = _response()
        user_only = [{"role": "user", "content": "hi"}]

        await c.generate(user_only)

        assert _sent(c.client.chat.completions.create.call_args) == user_only
        assert c._cache_marker_supported is None  # nothing was tested

    async def test_switched_off_sends_the_one_joined_string(self):
        """No marker → the request is exactly the pre-split one: one user string."""
        c = _make_client(prompt_cache=False)
        c.client.chat.completions.create.return_value = _response()

        await c.generate(MESSAGES)

        assert _sent(c.client.chat.completions.create.call_args) == JOINED
        assert GLOBAL_STATS.cache_markers() == {MODEL: "off"}

    async def test_marker_state_follows_each_run_not_the_first(self):
        """on → off → on in one process: the record says what the latest run did."""
        on = _make_client()
        on.client.chat.completions.create.return_value = _response()
        await on.generate(MESSAGES)
        off = _make_client(prompt_cache=False)
        off.client.chat.completions.create.return_value = _response()
        await off.generate(MESSAGES)
        assert GLOBAL_STATS.cache_markers() == {MODEL: "off"}
        again = _make_client()                     # learns "accepted" from the shared cache
        again.client.chat.completions.create.return_value = _response()
        await again.generate(MESSAGES)
        assert GLOBAL_STATS.cache_markers() == {MODEL: "sent"}

    async def test_plain_prompt_is_marked_too(self):
        """Level 5 sends the caller's plain prompt — it carries the marker as well."""
        class Schema(BaseModel):
            ok: bool

        c = _make_client()
        c._strategy_index = len(RETRY_MATRIX) - 1  # 'Unguided Decoding'
        c.client.chat.completions.create.return_value = _response()
        plain = _two_blocks("Plain fixed. ", "x")

        await c.generate(MESSAGES, schema=Schema, plain_messages=plain)

        sent = _sent(c.client.chat.completions.create.call_args)
        assert sent[1]["content"][0] == {"type": "text", "text": "Plain fixed. ", "cache_control": MARKER}

    async def test_litellm_path_sends_the_same_marker(self):
        c = LLMClient(api_key="k", model="openrouter/some/model", base_url=None)
        c._connection_verified = True
        c._strategy_discovered = True
        c._discovery_succeeded = True
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_response()) as acompletion:
            await c.generate(MESSAGES)

        assert _is_marked(acompletion.call_args.kwargs["messages"])


# ── The probe: the marker can never break a run ──────────────────


@pytest.mark.integration
class TestMarkerProbe:

    async def test_accepted_marker_is_learned(self):
        c = _make_client()
        c.client.chat.completions.create.return_value = _response()

        await c.generate(MESSAGES)

        assert c._cache_marker_supported is True
        assert LLMClient._CACHE_MARKER_CACHE[(BASE_URL, MODEL)] is True
        assert GLOBAL_STATS.cache_markers() == {MODEL: "sent"}

    async def test_rejected_marker_is_dropped_and_the_item_survives(self, caplog):
        c = _make_client()
        c.client.chat.completions.create.side_effect = [
            _bad_request("unknown field: cache_control"),
            _response(),
        ]

        with caplog.at_level(logging.INFO):
            out = await c.generate(MESSAGES)

        assert out == '{"ok": true}'
        first, second = c.client.chat.completions.create.call_args_list
        assert _is_marked(_sent(first))
        assert _sent(second) == JOINED                 # same request, as the one joined string
        assert c._cache_marker_supported is False
        assert "rejected the prompt-cache marker" in caplog.text
        assert GLOBAL_STATS.cache_markers() == {MODEL: "rejected"}

    async def test_rejection_is_shared_so_no_client_sends_it_again(self):
        c1 = _make_client()
        c1.client.chat.completions.create.side_effect = [_bad_request("cache_control"), _response()]
        await c1.generate(MESSAGES)

        c2 = _make_client()
        c2.client.chat.completions.create.return_value = _response()
        await c2.generate(MESSAGES)

        assert c2.client.chat.completions.create.call_count == 1
        assert _sent(c2.client.chat.completions.create.call_args) == JOINED

    async def test_innocent_marker_does_not_hide_the_real_capability_gap(self):
        """Discovery: marked 400 → unmarked 400 too → the marker was not the cause.
        drop_params and the matrix then run exactly as before, and the marker
        stays unknown (so it is tried again on the next call)."""
        c = _make_client(discovered=False)  # strategy 0: reasoning + drop_params
        c.client.chat.completions.create.side_effect = [
            _bad_request("Unsupported parameter(s): `drop_params`"),  # marked
            _bad_request("Unsupported parameter(s): `drop_params`"),  # probe: unmarked, still 400
            _response(),                                              # drop_params off
        ]

        out = await c.generate(MESSAGES)

        assert out == '{"ok": true}'
        calls = c.client.chat.completions.create.call_args_list
        assert [_is_marked(_sent(call)) for call in calls] == [True, False, False]
        assert c._drop_params_supported is False
        assert c._cache_marker_supported is None
        assert c._strategy_index == 0

    async def test_non_400_errors_never_trigger_the_probe(self):
        from openai import InternalServerError
        resp = MagicMock(status_code=500, headers={})
        c = _make_client()
        c.client.chat.completions.create.side_effect = [
            InternalServerError(message="boom", response=resp, body=None),
            _response(),
        ]
        with patch("claimlens.llmclient.asyncio.sleep", new_callable=AsyncMock):
            await c.generate(MESSAGES)

        calls = c.client.chat.completions.create.call_args_list
        assert all(_is_marked(_sent(call)) for call in calls)
        assert c._cache_marker_supported is True


# ── What the user sees while the calls run ───────────────────────


@pytest.mark.integration
class TestLiveLines:

    async def _answers(self, c, cached_values, task="extract"):
        c.client.chat.completions.create.side_effect = [_response(v) for v in cached_values]
        for _ in cached_values:
            await c.generate(MESSAGES, task=task)

    async def test_first_hit_is_announced_once_without_a_share(self, caplog):
        c = _make_client()
        with caplog.at_level(logging.INFO):
            await self._answers(c, [0, 900, 900, 900])
        assert caplog.text.count("Prompt cache hit for test-model (extract)") == 1
        assert "%" not in caplog.text

    async def test_not_hitting_warns(self, caplog):
        c = _make_client()
        with caplog.at_level(logging.INFO):
            await self._answers(c, [0] * CACHE_CHECK_AFTER)
        assert (f"Prompt cache not hitting for test-model (extract): none of the first "
                f"{CACHE_CHECK_AFTER} requests") in caplog.text

    async def test_not_reported_is_said_once_on_the_first_answer(self, caplog):
        c = _make_client()
        with caplog.at_level(logging.INFO):
            await self._answers(c, [None, None])
        assert caplog.text.count("test-model (extract): endpoint does not report cache use") == 1

    async def test_misses_before_the_check_point_stay_quiet(self, caplog):
        c = _make_client()
        with caplog.at_level(logging.INFO):
            await self._answers(c, [0] * (CACHE_CHECK_AFTER - 1))
        assert "hit" not in caplog.text and "not report" not in caplog.text

    async def test_stopped_hitting_warns_once_per_streak(self, caplog):
        c = _make_client()
        with caplog.at_level(logging.INFO):
            await self._answers(c, [900] * CACHE_CHECK_AFTER + [0] * (CACHE_CHECK_AFTER + 5))
        assert caplog.text.count("Prompt cache stopped hitting for test-model (extract)") == 1

    async def test_tasks_are_judged_separately(self, caplog):
        c = _make_client()
        with caplog.at_level(logging.INFO):
            await self._answers(c, [900] * CACHE_CHECK_AFTER, task="extract")
            await self._answers(c, [0] * CACHE_CHECK_AFTER, task="check")
        assert "Prompt cache hit for test-model (extract)" in caplog.text
        assert "not hitting for test-model (check)" in caplog.text

    async def test_size_note_once_per_task(self, caplog):
        c = _make_client()
        with caplog.at_level(logging.INFO):
            await self._answers(c, [0, 0])
        assert caplog.text.count("Prompt cache on for test-model (extract) · fixed prompt part ~6 tokens "
                                 "(below 1,024 — most providers won't cache it)") == 1

    async def test_size_above_the_minimum_is_debug_only(self, caplog):
        c = _make_client()
        c.client.chat.completions.create.return_value = _response(0)
        big = _two_blocks("x" * 8000, ITEM)
        with caplog.at_level(logging.INFO):
            await c.generate(big, task="extract")
        assert "fixed prompt part" not in caplog.text
        LLMClient._CACHE_WATCH.clear()
        with caplog.at_level(logging.DEBUG):
            await c.generate(big, task="extract")
        assert "fixed prompt part ~2,001 tokens (above 1,024)" in caplog.text

    async def test_a_failing_watch_never_breaks_the_request(self):
        """The live lines are display only: if they fail, the answer still comes back."""
        c = _make_client()
        c.client.chat.completions.create.return_value = _response(900)

        with patch.object(c, "_watch_cache", side_effect=RuntimeError("display bug")):
            assert await c.generate(MESSAGES, task="extract") == '{"ok": true}'

    async def test_switched_off_prints_no_cache_lines(self, caplog):
        c = _make_client(prompt_cache=False)
        with caplog.at_level(logging.INFO):
            await self._answers(c, [900] * CACHE_CHECK_AFTER)
        assert "Prompt cache" not in caplog.text


# ── The CLI switch ───────────────────────────────────────────────


@pytest.mark.integration
class TestCliFlag:
    """--prompt-cache/--no-prompt-cache reaches the service like --concurrency."""

    def _invoke(self, tmp_path, monkeypatch, *extra) -> dict:
        from typer.testing import CliRunner
        from claimlens.cli import app
        seen = {}

        class _Service:
            def __init__(self, **kwargs):
                seen.update(kwargs)

            def run_sync(self, data):
                return data

        monkeypatch.setattr("claimlens.services.extraction.ExtractionService", _Service)
        src = tmp_path / "d.json"
        src.write_text('[{"response": "r"}]', encoding="utf-8")
        result = CliRunner().invoke(app, ["extract", str(src), "-m", "m", "-o", str(tmp_path / "o.json"), *extra])
        assert result.exit_code == 0, result.output
        return seen

    def test_on_by_default(self, tmp_path, monkeypatch):
        assert self._invoke(tmp_path, monkeypatch)["prompt_cache"] is True

    def test_no_prompt_cache_switches_it_off(self, tmp_path, monkeypatch):
        assert self._invoke(tmp_path, monkeypatch, "--no-prompt-cache")["prompt_cache"] is False

    def test_worker_hands_it_to_the_client(self):
        from claimlens.workers.extractor import Extractor
        assert Extractor(api_key="k", model="m", base_url=BASE_URL, prompt_cache=False).client.prompt_cache is False
