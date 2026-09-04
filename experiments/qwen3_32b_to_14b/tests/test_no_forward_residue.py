from __future__ import annotations

from pathlib import Path
import re


def test_no_obsolete_reverse_dimensions_in_text_files() -> None:
    root = Path(__file__).resolve().parents[1]
    # 8192 is both the obsolete k=8 mapper feature width and a legitimate
    # paper latency sequence length.  Only reject it when the surrounding
    # field/shape makes the obsolete meaning unambiguous.
    obsolete_contexts = (
        re.compile(
            r"\b(?:source_features|input_features|in_features)\b"
            r"[^\n]{0,48}(?<!\d)8_?192(?!\d)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?<!\d)8_?192(?!\d)[^\n]{0,24}"
            r"(?:x|\u00d7|->|\u2192)\s*1_?024\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:selected_source_layers|top[_ -]?k)\b"
            r"[^\n]{0,32}(?:[:=]\s*8\b|\bk8\b)",
            re.IGNORECASE,
        ),
        re.compile(r"\bk8\b", re.IGNORECASE),
        re.compile(r"\b2\.50\s*GiB\b", re.IGNORECASE),
    )
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md", ".yaml", ".json"}:
            continue
        # Tamper tests intentionally construct obsolete values to exercise the
        # fail-closed gates; they are not protocol residue.
        if "tests" in path.relative_to(root).parts:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in obsolete_contexts:
            assert pattern.search(text) is None, (
                f"obsolete mapper context {pattern.pattern!r} remains in {path}"
            )
