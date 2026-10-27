"""docs/CONVENTIONS.md mirrors the il-conventions skill verbatim (frontmatter stripped)."""

from __future__ import annotations

from ila import p8_adapter

SKILL = p8_adapter.REPO_ROOT / ".claude" / "skills" / "il-conventions" / "SKILL.md"
DOC = p8_adapter.REPO_ROOT / "docs" / "CONVENTIONS.md"


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.index("\n---\n", 4)
    return text[end + len("\n---\n") :].lstrip("\n")


def test_conventions_doc_equals_skill_body() -> None:
    skill = _strip_frontmatter(SKILL.read_text(encoding="utf-8"))
    doc = DOC.read_text(encoding="utf-8")
    # Leading and trailing blank lines are not content (the doc starts with one).
    assert doc.strip("\n") == skill.strip("\n")
