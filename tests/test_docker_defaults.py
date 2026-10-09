"""The Docker build defaults install this checkout's release, not a stale one."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["version"]


def test_dockerfile_hf_version_default_matches_pyproject():
    match = re.search(r"^ARG HF_VERSION=(\S+)$", (ROOT / "Dockerfile").read_text("utf-8"), re.M)
    assert match and match.group(1) == _version()


def test_compose_hf_version_matches_pyproject():
    versions = re.findall(r'HF_VERSION: "([^"]+)"', (ROOT / "docker-compose.yaml").read_text("utf-8"))
    assert versions and set(versions) == {_version()}
