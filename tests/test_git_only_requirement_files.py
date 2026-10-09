"""requirements/{sf3d,triposr,trellis2}.txt must be installable as written.

They pointed `pip install -r` at the roots of stable-fast-3d, TripoSR and
TRELLIS.2, none of which has a setup.py or pyproject.toml at any commit, so
pip failed with "does not appear to be a Python project". Packaged
subdirectories (texture_baker, uv_unwrapper) are fine; the roots are not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
UNPACKAGED_ROOTS = (
    "github.com/Stability-AI/stable-fast-3d",
    "github.com/VAST-AI-Research/TripoSR",
    "github.com/microsoft/TRELLIS.2",
)


def _requirement_lines(name: str) -> list[str]:
    text = (ROOT / "requirements" / f"{name}.txt").read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize("name", ["sf3d", "triposr", "trellis2"])
def test_no_requirement_points_at_an_unpackaged_repository_root(name):
    for line in _requirement_lines(name):
        requirement = Requirement(line)
        url = requirement.url or ""
        if any(root in url for root in UNPACKAGED_ROOTS):
            subdirectory = url.partition("#subdirectory=")[2]
            assert subdirectory not in ("", "."), (
                f"{name}.txt installs {url}, a repository root with no Python "
                "packaging; pip cannot install it"
            )


def test_files_inspected_something():
    assert _requirement_lines("sf3d") and _requirement_lines("triposr")
