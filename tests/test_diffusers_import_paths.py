"""Every `from diffusers...` import in the package must actually resolve.

diffusers moves pipelines between releases and deletes the deep module paths.
Three had already rotted without anything failing at import time, because each
sits inside a method and only raises when the node runs:

* `diffusers.pipelines.audioldm.pipeline_audioldm` — gone; `AudioLDMPipeline`
  is exported at the top level.
* `diffusers.pipelines.musicldm.pipeline_musicldm` — same.
* `diffusers.pipelines.wan.pipeline_wan_flf2v` — gone with no top-level
  equivalent: upstream folded first-last-frame into `WanImageToVideoPipeline`,
  which takes `last_image`.

MusicLDM surfaced on a GPU sweep as "No module named
'diffusers.pipelines.musicldm'" after the model had downloaded. This test
finds the whole class before a user pays for a download.

It skips when the installed diffusers is below the floor in pyproject.toml. An
older release is not evidence: `encode_video` is absent from diffusers 0.38 and
present in 0.40, so an env under the pin reports a symbol as rotted when it is
simply newer than the env.
"""

import ast
import importlib
import re
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _diffusers_imports() -> list[tuple[str, str, str]]:
    """Every (module, symbol, source file) imported from diffusers."""
    found: list[tuple[str, str, str]] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module != "diffusers" and not node.module.startswith("diffusers."):
                continue
            for alias in node.names:
                found.append((node.module, alias.name, str(path.relative_to(SRC))))
    return sorted(set(found))


IMPORTS = _diffusers_imports()


def test_the_scan_found_imports():
    """A scan that matches nothing would pass every case below silently."""
    assert len(IMPORTS) > 20, f"only found {len(IMPORTS)} diffusers imports"


def _diffusers_below_floor() -> bool:
    """Is the installed diffusers older than the floor pyproject declares?"""
    diffusers = pytest.importorskip("diffusers")
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    match = re.search(r'diffusers\[torch\]>=([0-9]+(?:\.[0-9]+)*)', text)
    if not match:
        return False
    def parts(version: str) -> list[int]:
        return [int(p) for p in re.findall(r"[0-9]+", version)]

    return parts(diffusers.__version__) < parts(match.group(1))


@pytest.mark.parametrize(
    "module, symbol, source",
    IMPORTS,
    ids=[f"{m}.{s}" for m, s, _ in IMPORTS],
)
def test_diffusers_import_resolves(module, symbol, source):
    pytest.importorskip("diffusers")
    if _diffusers_below_floor():
        pytest.skip("installed diffusers is below the floor in pyproject.toml")
    try:
        loaded = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"{source} imports `from {module} import {symbol}`, but the module "
            f"is gone in diffusers {importlib.import_module('diffusers').__version__}: {exc}"
        )
    assert hasattr(loaded, symbol), (
        f"{source} imports `{symbol}` from {module}, which no longer exports it"
    )
