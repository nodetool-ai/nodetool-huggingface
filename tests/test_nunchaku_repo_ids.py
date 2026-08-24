"""Guards on the nunchaku model repo ids this package names.

The HuggingFace organisation ``nunchaku-tech`` was renamed to ``nunchaku-ai``.
The Hub answers the old name with a rename redirect that carries no ETag, and
NodeTool's downloader fails on it, so every quantized FLUX and Qwen path breaks.
These tests are offline: they read the source tree, never the Hub.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every nunchaku repo id this package is allowed to name, plus the project's own
# GitHub/package-manager slug. Adding a model means adding it here on purpose.
KNOWN_NUNCHAKU_IDS = {
    "nunchaku-ai/nunchaku",
    "nunchaku-ai/nunchaku-flux.1-canny-dev",
    "nunchaku-ai/nunchaku-flux.1-depth-dev",
    "nunchaku-ai/nunchaku-flux.1-dev",
    "nunchaku-ai/nunchaku-flux.1-fill-dev",
    "nunchaku-ai/nunchaku-flux.1-kontext-dev",
    "nunchaku-ai/nunchaku-flux.1-schnell",
    "nunchaku-ai/nunchaku-qwen-image",
    "nunchaku-ai/nunchaku-qwen-image-edit",
    "nunchaku-ai/nunchaku-qwen-image-edit-2509",
    "nunchaku-ai/nunchaku-sdxl",
    "nunchaku-ai/nunchaku-sdxl-turbo",
    "nunchaku-ai/nunchaku-t5",
}

# ``FluxControl`` builds its transformer repo id from a variant key, so the
# literal in the source is a prefix rather than a whole id.
KNOWN_NUNCHAKU_PREFIXES = {"nunchaku-ai/nunchaku-flux.1-"}

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

STALE_ORG = "nunchaku-tech"
NUNCHAKU_ID_RE = re.compile(r"nunchaku-ai/[A-Za-z0-9._-]*")


def _text_files():
    """Every readable text file in the repo, except this guard itself.

    This file names the old organisation on purpose — in the constant below and
    in the failure message — so scanning it would make the guard fail against
    itself.
    """
    self_path = Path(__file__).resolve()
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.resolve() == self_path:
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        try:
            yield path, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def test_no_stale_nunchaku_org_references():
    """No file may still name the old organisation."""
    offenders = []
    scanned = 0
    for path, text in _text_files():
        scanned += 1
        for lineno, line in enumerate(text.splitlines(), start=1):
            if STALE_ORG in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")

    assert scanned > 50, f"scan reached only {scanned} files — the walk is broken"
    assert not offenders, (
        f"{len(offenders)} reference(s) to the renamed organisation "
        f"'{STALE_ORG}/' survive; use 'nunchaku-ai/':\n" + "\n".join(offenders)
    )


def test_every_declared_nunchaku_id_is_known():
    """Every nunchaku repo id named anywhere is one we pinned on purpose."""
    found: dict[str, set[str]] = {}
    for path, text in _text_files():
        for match in NUNCHAKU_ID_RE.findall(text):
            found.setdefault(match, set()).add(str(path.relative_to(REPO_ROOT)))

    assert found, "found no nunchaku repo ids at all — the scan is broken"

    unknown = {
        repo_id: sorted(paths)
        for repo_id, paths in found.items()
        if repo_id not in KNOWN_NUNCHAKU_IDS
        and repo_id not in KNOWN_NUNCHAKU_PREFIXES
    }
    assert not unknown, (
        "unknown nunchaku repo id(s); fix the typo or add them to "
        f"KNOWN_NUNCHAKU_IDS: {unknown}"
    )


def test_known_nunchaku_ids_are_well_formed():
    for repo_id in KNOWN_NUNCHAKU_IDS:
        org, _, name = repo_id.partition("/")
        assert org == "nunchaku-ai", repo_id
        assert name and name == name.strip(), repo_id
        assert re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name), repo_id


def test_flux_control_variants_resolve_to_known_ids():
    """The one repo id built at runtime stays inside the pinned set."""
    if __package__ is None or __package__ == "":
        sys.path.insert(0, str(REPO_ROOT))

    from nodetool.nodes.huggingface.text_to_image import (
        FluxControl,
        FluxControlQuantization,
    )

    for base_repo in (
        "black-forest-labs/FLUX.1-Canny-dev",
        "black-forest-labs/FLUX.1-Depth-dev",
    ):
        node = FluxControl()
        node.model.repo_id = base_repo
        _, transformer_model, text_encoder_model = node._resolve_model_config(
            FluxControlQuantization.INT4
        )
        assert transformer_model is not None
        assert transformer_model.repo_id in KNOWN_NUNCHAKU_IDS, transformer_model.repo_id
        assert text_encoder_model is not None
        assert text_encoder_model.repo_id in KNOWN_NUNCHAKU_IDS
