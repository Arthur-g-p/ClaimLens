"""
HTML viewer — one self-contained page rendered from a report.

The record already carries everything the console prints; the page is a
renderer for it, never a second source of numbers. Dispatch is on
``_meta.report_type``: a report type is viewable when
``templates/<report_type>.html`` exists, so a viewer for a new type is that
one file. The template gets the shared CSS and JS inlined and both documents
embedded as JSON script tags, so the page opens from file:// with no server,
no network and no dependencies.
"""

import json
from pathlib import Path

from claimlens.exceptions import ViewerError
from claimlens.utils import REPORT_SCHEMA_VERSION

TEMPLATES = Path(__file__).parent / "templates"


def supported_report_types() -> list[str]:
    """Report types that have a template."""
    return sorted(p.stem for p in TEMPLATES.glob("*.html"))


def render_html(record: dict, findings: dict | None = None) -> str:
    """The viewer page for *record*, with *findings* embedded when given.

    Raises ViewerError when the report type has no template or the record's
    schema_version is not the one the templates read.
    """
    meta = record.get("_meta") or {}
    report_type = meta.get("report_type")
    template = TEMPLATES / f"{report_type}.html" if report_type else None
    if template is None or not template.exists():
        raise ViewerError(
            f"No HTML viewer for report type {report_type!r}. "
            f"Viewable: {', '.join(supported_report_types())}."
        )
    version = meta.get("schema_version")
    if version != REPORT_SCHEMA_VERSION:
        raise ViewerError(
            f"Report schema_version {version!r} does not match this viewer "
            f"({REPORT_SCHEMA_VERSION}). Re-run the report with this claimlens version."
        )

    page = template.read_text(encoding="utf-8")
    page = page.replace("/*COMMON_CSS*/", (TEMPLATES / "common.css").read_text(encoding="utf-8"))
    page = page.replace("/*COMMON_JS*/", (TEMPLATES / "common.js").read_text(encoding="utf-8"))
    return page.replace("__RECORD__", _embed(record)).replace("__FINDINGS__", _embed(findings or {}))


def _embed(document: dict) -> str:
    # A "<" inside any string could close the script tag; as < it is still the same JSON.
    return json.dumps(document, ensure_ascii=False).replace("<", "\\u003c")
