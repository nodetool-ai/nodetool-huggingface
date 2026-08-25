"""Regression: TripoSG reserved 3 GB of disk for a 7.95 GB download.

`_check_disk_space(estimated_gb)` raises when the HuggingFace cache volume has
less than ``estimated_gb + _DISK_HEADROOM_GB`` free. TripoSG passed
``ESTIMATED_DOWNLOAD_GB = 3.0``, so on a volume with, say, 6 GB free the check
passed and `snapshot_download` then ran out of space partway through a download
2.6x larger than what was reserved. Under-reserving disk is the defect; the
docstring's "~3GB" was the visible symptom of the same wrong number.

Both figures are read from the Hub's own blob sizes:

    VAST-AI/TripoSG   11 files   7,946,494,238 B   7.95 GB / 7.40 GiB
    briaai/RMBG-1.4   20 files     842,215,095 B   0.84 GB / 0.78 GiB

Neither `snapshot_download` call passes `allow_patterns`, so each fetches its
whole repo. RMBG publishes its weights in five formats (.pth, .bin,
.safetensors, and three ONNX variants); the node loads one 0.18 GB file, which
is where the old `# RMBG is ~0.2 GB` comment came from, but the download takes
all twenty files.

The two reservations stay separate rather than being summed: each sits directly
before its own `snapshot_download`, and the second runs after the first has
already consumed disk, so `_check_disk_space` sees the reduced free space at the
moment it matters.

No network here -- the sizes above are constants, and the check is driven
against a stubbed volume.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT.parent / "nodetool-core" / "src"
HF_SRC = ROOT / "src"
for _p in (str(CORE_SRC), str(HF_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nodetool.nodes.huggingface import _3d_common  # noqa: E402
from nodetool.nodes.huggingface.image_to_3d import TripoSG  # noqa: E402

IMAGE_TO_3D = HF_SRC / "nodetool" / "nodes" / "huggingface" / "image_to_3d.py"

# Hub blob totals, in bytes.
TRIPOSG_BYTES = 7_946_494_238
RMBG_BYTES = 842_215_095
GIB = 1 << 30


@pytest.fixture
def volume_with(monkeypatch):
    """Pretend the HF cache volume has exactly `free_gib` GiB free."""
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")

    def _set(free_gib: float):
        monkeypatch.setattr(
            shutil, "disk_usage", lambda _p: usage(0, 0, int(free_gib * GIB))
        )

    return _set


def _needs(estimated_gb: float) -> float:
    """What `_check_disk_space` demands free, in GiB."""
    return estimated_gb + _3d_common._DISK_HEADROOM_GB


# ---------------------------------------------------------------------------
# the defect: the reservation must cover the download
# ---------------------------------------------------------------------------


def test_triposg_reservation_covers_the_real_download(volume_with, tmp_path):
    """A volume that cannot fit TripoSG must be refused, not accepted."""
    real_gib = TRIPOSG_BYTES / GIB

    # Enough for the old 3.0 reservation, nowhere near enough for the download.
    volume_with(real_gib - 1)
    with pytest.raises(OSError):
        _3d_common._check_disk_space(
            TripoSG.ESTIMATED_DOWNLOAD_GB, cache_dir=str(tmp_path)
        )


def test_rmbg_reservation_covers_its_whole_repo(volume_with, tmp_path):
    """The node loads 0.18 GB but snapshot_download fetches 0.84 GB."""
    real_gib = RMBG_BYTES / GIB
    reserved = _rmbg_reservation()

    volume_with(_needs(real_gib) - 0.01)
    with pytest.raises(OSError):
        _3d_common._check_disk_space(reserved, cache_dir=str(tmp_path))


def test_a_volume_that_fits_everything_is_accepted(volume_with, tmp_path):
    """The guard must not refuse a machine that can actually do the job."""
    volume_with(_needs(TripoSG.ESTIMATED_DOWNLOAD_GB) + 1)
    _3d_common._check_disk_space(
        TripoSG.ESTIMATED_DOWNLOAD_GB, cache_dir=str(tmp_path)
    )  # must not raise


# ---------------------------------------------------------------------------
# the constants themselves
# ---------------------------------------------------------------------------


def test_estimated_download_clears_the_real_size_in_either_unit():
    """`_check_disk_space` compares against free GiB despite the _GB name."""
    assert TripoSG.ESTIMATED_DOWNLOAD_GB >= TRIPOSG_BYTES / GIB, (
        "the reservation is below the download size in GiB, the unit "
        "_check_disk_space actually compares"
    )
    assert TripoSG.ESTIMATED_DOWNLOAD_GB >= TRIPOSG_BYTES / 1e9, (
        "the reservation is below the download size in GB either"
    )


def _rmbg_reservation() -> float:
    """The literal TripoSG._load_models passes to _check_disk_space for RMBG."""
    tree = ast.parse(IMAGE_TO_3D.read_text(encoding="utf-8"))
    triposg = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "TripoSG"
    )
    load_models = next(
        n
        for n in ast.walk(triposg)
        if isinstance(n, ast.FunctionDef) and n.name == "_load_models"
    )
    literals = [
        node.args[0].value
        for node in ast.walk(load_models)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_check_disk_space"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    assert len(literals) == 1, f"expected one literal RMBG reservation, got {literals}"
    return float(literals[0])


def test_rmbg_reservation_clears_its_real_size():
    reserved = _rmbg_reservation()
    assert reserved >= RMBG_BYTES / GIB
    assert reserved >= RMBG_BYTES / 1e9


def test_docstring_does_not_understate_the_first_run_download():
    """The docstring is what a user reads before committing the bandwidth."""
    doc = TripoSG.__doc__ or ""
    assert "~3GB" not in doc and "~3 GB" not in doc, (
        "the docstring still claims ~3GB; the real first run is ~8.8 GB"
    )
    total_gb = (TRIPOSG_BYTES + RMBG_BYTES) / 1e9
    assert f"{total_gb:.1f}" in doc.replace(" GB", "GB"), (
        f"the docstring should state the real first-run total (~{total_gb:.1f} GB)"
    )


def test_both_downloads_are_reserved_for_separately():
    """Each snapshot_download is preceded by its own check."""
    source = IMAGE_TO_3D.read_text(encoding="utf-8")
    tree = ast.parse(source)
    triposg = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "TripoSG"
    )
    load_models = next(
        n
        for n in ast.walk(triposg)
        if isinstance(n, ast.FunctionDef) and n.name == "_load_models"
    )
    checks = [
        n.lineno
        for n in ast.walk(load_models)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_check_disk_space"
    ]
    downloads = [
        n.lineno
        for n in ast.walk(load_models)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "snapshot_download"
    ]
    assert len(checks) == len(downloads) == 2
    for check, download in zip(sorted(checks), sorted(downloads)):
        assert check < download, "a download runs before its disk check"
