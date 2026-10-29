"""
Prompt map — the prompts every worker sends, and how a request is built from them.

Every prompt is {"system": ..., "instructions": ..., "input": ...}: the system
message, the fixed task (the same on every call), and the current case with
the item placeholders. The shipped map is prompt_map.json; --prompts lays a
user file over it. The entry point (CLI or library verb) resolves the file once
and everything below receives the resulting map, passed down like concurrency.

The request built from a prompt (prompt_messages) is the counterpart of
LLMClient's cache marker (llmclient._with_cache_marker): the marker goes on the
instructions block, so everything before the item can be cached.

This file is a leaf dependency — it imports nothing from claimlens.
"""

import hashlib
import json
import logging
import re
from pathlib import Path


PROMPT_PATH: Path = Path(__file__).parent / "prompt_map.json"

PromptMap = dict[str, dict[str, str]]
"""{key: {"system": ..., "instructions": ..., "input": ...}}."""

PARTS = ("system", "instructions", "input")

REQUIRED_KEYS = (
    "extractor_prompt", "extractor_prompt_plain",
    "checker_prompt", "checker_prompt_plain",
    "checker_prompt_joint", "checker_prompt_joint_plain",
    "checker_prompt_eval_joint", "checker_prompt_eval_joint_plain",
    "atomizer_prompt", "atomizer_prompt_plain",
)
"""Every key the code reads. The shipped map must hold all of them; a --prompts
file holds any subset, the shipped prompt filling each key it leaves out."""

_PLACEHOLDER = re.compile(r"\{\{\w+\}\}")
_ITEM_PLACEHOLDER = re.compile(r"\{\{(?!schema\}\})\w+\}\}")
"""An item placeholder ({{text}}, {{claims}}, ...). {{schema}} is not one: it is
filled once at start, so the instructions may carry it."""


def _check_prompt(label: str, key: str, prompt) -> None:
    """Raise ValueError unless *prompt* is {system, instructions, input}, all
    strings, with no placeholder in the system message and no item placeholder
    in the instructions."""
    if not (isinstance(prompt, dict) and set(prompt) == set(PARTS)
            and all(isinstance(v, str) for v in prompt.values())):
        raise ValueError(f'{label}: "{key}" must be '
                         f'{{"system": "...", "instructions": "...", "input": "..."}}')
    if _PLACEHOLDER.search(prompt["system"]):
        raise ValueError(f'{label}: "{key}" has a placeholder in its system message')
    if _ITEM_PLACEHOLDER.search(prompt["instructions"]):
        raise ValueError(f'{label}: "{key}" has an item placeholder in its instructions; '
                         f"it belongs in input, the part that changes per call")


def _load_shipped() -> PromptMap:
    """Load prompt_map.json shipped with the package."""
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"prompt_map.json not found at {PROMPT_PATH}. Package may be corrupted."
        )
    try:
        prompts = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"prompt_map.json is not valid JSON: {exc}") from exc
    missing = [k for k in REQUIRED_KEYS if k not in prompts]
    if missing:
        raise ValueError(f"prompt_map.json lacks required prompt(s): {', '.join(missing)}")
    for key, prompt in prompts.items():
        _check_prompt("prompt_map.json", key, prompt)
    return prompts


PROMPTS: PromptMap = _load_shipped()


def resolve_prompts(path: str | Path | None = None) -> PromptMap:
    """The full prompt map for a run: the shipped prompts with a prompt file laid over them.

    None = the shipped prompt_map.json as is. The file (--prompts) holds only the
    keys it changes, each complete. A broken file, an unknown key, a wrong shape,
    a placeholder where none belongs or that nothing fills, or an input missing
    one the shipped input has raise ValueError, so a bad file stops the run
    before any request.
    """
    if path is None:
        return PROMPTS
    if not isinstance(path, (str, Path)):
        raise ValueError(f"prompts= takes the path of a prompt file, not {type(path).__name__}")
    path = Path(path)
    label = str(path)
    try:
        overrides = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Prompt file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prompt file {path} is not valid JSON: {exc}") from exc
    if not isinstance(overrides, dict):
        raise ValueError(f"{label}: expected a JSON object of {{key: {{system, instructions, input}}}}")
    unknown = sorted(set(overrides) - set(PROMPTS))
    if unknown:
        raise ValueError(f"{label}: unknown prompt key(s) {', '.join(unknown)}. "
                         f"Known keys: {', '.join(sorted(PROMPTS))}")
    for key, prompt in overrides.items():
        _check_prompt(label, key, prompt)
        shipped = PROMPTS[key]
        missing = (set(_ITEM_PLACEHOLDER.findall(shipped["input"]))
                   - set(_ITEM_PLACEHOLDER.findall(prompt["input"])))
        if missing:
            raise ValueError(f'{label}: "{key}" input lacks {", ".join(sorted(missing))}')
        # A placeholder the shipped prompt never fills would reach the model as literal text.
        unfilled = (set(_PLACEHOLDER.findall(prompt["instructions"] + prompt["input"]))
                    - set(_PLACEHOLDER.findall(shipped["instructions"] + shipped["input"])))
        if unfilled:
            raise ValueError(f'{label}: "{key}" has placeholder(s) nothing fills: {", ".join(sorted(unfilled))}')
    return {**PROMPTS, **overrides}


def prompt(templates: PromptMap, key: str) -> dict[str, str]:
    """The prompt *key* from a map, or a ValueError naming it — not a bare
    KeyError deep in a worker when a map built by hand lacks a key."""
    try:
        return templates[key]
    except KeyError:
        raise ValueError(f'the prompt map has no "{key}"') from None


def split_template(template: str, system: str) -> dict[str, str]:
    """One prompt map entry from a one-string template: the text before its first
    item placeholder is the instructions, the rest the input. Lossless —
    instructions + input is the template. Converts a one-string prompt (an
    older prompt map, a prompt built as one text) into the current shape."""
    match = _ITEM_PLACEHOLDER.search(template)
    cut = match.start() if match else len(template)
    return {"system": system, "instructions": template[:cut], "input": template[cut:]}


def prompts_overridden(templates: PromptMap) -> list[str]:
    """The keys whose prompt differs from the shipped one, for
    _meta.prompts_overridden: which prompts a run took from a --prompts file."""
    return sorted(k for k, v in templates.items() if PROMPTS.get(k) != v)


def prompts_sha256(templates: PromptMap) -> str:
    """Short hash of the prompt map a run used, for _meta.prompts_sha256: a record
    still names its prompts after the file changes. Content only — the same
    map hashes the same from any file name. Covers all three parts."""
    canonical = json.dumps(templates, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]


def describe_prompts(templates: PromptMap) -> str:
    """The Config block's Prompts line: shipped, or which keys are custom, and the hash."""
    overridden = prompts_overridden(templates)
    source = (f"custom {', '.join(overridden)}, over {PROMPT_PATH.name}" if overridden
              else str(PROMPT_PATH))
    return f"{source} · sha256 {prompts_sha256(templates)}"


def format_prompt(template: str, variables: dict) -> str:
    """Replace ``{{key}}`` placeholders in *template* with values from *variables*."""
    result = template
    for key, value in variables.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


_SCHEMA_SHAPE_MAX_DEPTH = 6


def build_schema_shape(schema) -> str:
    """Render a Pydantic model as the JSON shape a prompt should ask for."""
    def render(node: dict, defs: dict, depth: int, seen: frozenset):
        if depth > _SCHEMA_SHAPE_MAX_DEPTH:
            return "<...>"

        ref = node.get("$ref")
        if ref:
            name = ref.rsplit("/", 1)[-1]
            if name in seen:
                return "<recursive>"
            return render(defs.get(name, {}), defs, depth, seen | {name})

        for union_key in ("anyOf", "oneOf"):
            if union_key in node:
                branches = [b for b in node[union_key] if b.get("type") != "null"]
                return render(branches[0], defs, depth, seen) if branches else None

        # A one-value Literal arrives as const, not enum.
        if "const" in node:
            return str(node["const"])
        if "enum" in node:
            return " | ".join(str(v) for v in node["enum"])

        node_type = node.get("type")
        if node_type == "object" or "properties" in node:
            if "properties" in node:
                return {
                    key: render(sub, defs, depth + 1, seen)
                    for key, sub in node["properties"].items()
                }
            # A mapping carries its value type in additionalProperties instead.
            values = node.get("additionalProperties")
            return {"<key>": render(values, defs, depth + 1, seen)
                    if isinstance(values, dict) else "<any>"}
        if node_type == "array":
            # Fixed-length tuples enumerate their members instead of one item type.
            if "prefixItems" in node:
                return [render(i, defs, depth + 1, seen) for i in node["prefixItems"]]
            return [render(node.get("items", {}), defs, depth + 1, seen)]

        return f"<{node_type or 'string'}>"

    try:
        json_schema = schema.model_json_schema()
        shape = render(json_schema, json_schema.get("$defs", {}), 0, frozenset())
        return json.dumps(shape, indent=2)
    except Exception:
        return '{"result": "<see prompt for format>"}'


def prepare_plain_prompt(
    prompt: dict[str, str] | None, key: str, schema, logger: logging.Logger,
) -> dict[str, str] | None:
    """Substitute {{schema}} into a plain prompt's instructions and input.

    A missing prompt or a missing placeholder only matters if the endpoint
    needs unguided decoding, so both warn here and fail there. Logs through
    *logger* so the warning names the caller, not this module.
    """
    if prompt is None:
        logger.warning(
            "   ⚠️  No plain prompt '%s' in prompt_map — a model that cannot take "
            "a schema will abort the run.", key,
        )
        return None
    if "{{schema}}" not in prompt["instructions"] + prompt["input"]:
        logger.warning(
            "   ⚠️  Plain prompt '%s' has no {{schema}} placeholder — the model "
            "will get no output shape.", key,
        )
        return prompt
    shape = {"schema": build_schema_shape(schema)}
    return {part: format_prompt(text, shape) if part != "system" else text
            for part, text in prompt.items()}


def prompt_messages(prompt: dict[str, str], variables: dict) -> list[dict]:
    """System + user messages for one call, built from one prompt map entry.

    The user message is two text blocks: the instructions, then the input
    filled in for this item. LLMClient (llmclient._with_cache_marker) marks the
    second-to-last block of the last user message, which is the instructions,
    or joins the two back into one string when it sends no marker.
    """
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": [
            {"type": "text", "text": prompt["instructions"]},
            {"type": "text", "text": format_prompt(prompt["input"], variables)},
        ]},
    ]
