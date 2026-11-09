"""Script file names must not shadow a standard-library module.

``python scripts/<name>.py`` puts ``scripts/`` first on ``sys.path``. A script named like a stdlib
module (``select.py`` did) is then imported in its place on Pythons where that module is not built
into the interpreter (CI's hostedtoolcache Python), which broke ``make report`` there (P10-D2
item 8).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize(
    "path", sorted(SCRIPTS.rglob("*.py")), ids=lambda p: p.relative_to(SCRIPTS).as_posix()
)
def test_script_does_not_shadow_stdlib(path: Path) -> None:
    assert path.stem not in sys.stdlib_module_names, f"{path} shadows stdlib module {path.stem!r}"
