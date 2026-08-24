"""Regression: TripoSG raised a bare ModuleNotFoundError after a 7.9 GB download.

On a RunPod 4xA40 pod running `main`, `huggingface.image_to_3d.TripoSG` failed
with::

    No module named 'skimage'

while its three siblings -- Hunyuan3D, StableFast3D, Trellis2 -- all catch the
ImportError and name what to install. The dependency *declaration* is fine:
scikit-image is in the `triposg` extra, alongside opencv-python and pymeshlab.
What was missing is the guard.

Where it failed matters as much as that it failed. `TripoSG._prepare_image`
imports cv2 and skimage, and it runs *after* `_load_models()` has downloaded
`VAST-AI/TripoSG`. That `snapshot_download` call passes no `allow_patterns`, so
it pulls the whole repo -- 7.95 GB by the Hub's own blob sizes -- on top of
`briaai/RMBG-1.4`. An unguarded import there spends all of that bandwidth and
*then* discovers the node cannot run. So the guard belongs before the download,
which is the failure point #71 and #76 established.

TripoSG already guarded `diso` this way, so this completes a pattern its own
author started rather than introducing one.

Every check here drives the node's public surface -- `preload_model` and
`process` -- with the modules genuinely blocked at the import system, rather
than reaching for an internal helper. That keeps them behavioural: on
unmodified `main` they fail because the node downloads and then dies, not
because a symbol is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT.parent / "nodetool-core" / "src"
HF_SRC = ROOT / "src"
for _p in (str(CORE_SRC), str(HF_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nodetool.metadata.types import ImageRef  # noqa: E402
from nodetool.nodes.huggingface import image_to_3d  # noqa: E402
from nodetool.nodes.huggingface._3d_common import MissingDependencyError  # noqa: E402

# Everything TripoSG imports that a base install may not have. All three ship
# in the [triposg] extra. The triposg package itself is NOT here: it is vendored
# at src/triposg and ships inside the wheel, so it is never a user's to install
# -- test_vendored_triposg_is_part_of_the_wheel pins that.
EXTRA_MODULES = ("cv2", "skimage", "pymeshlab")


class _BlockImports:
    """A meta_path finder that makes named top-level modules unimportable."""

    def __init__(self, names: set[str]):
        self.names = names

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.names:
            raise ModuleNotFoundError(f"No module named {root!r}", name=root)
        return None


@pytest.fixture
def block_modules(monkeypatch):
    """Make the named modules genuinely absent, cache included."""

    def _block(*names: str):
        blocked = set(names)
        for cached in list(sys.modules):
            if cached.split(".")[0] in blocked:
                monkeypatch.delitem(sys.modules, cached, raising=False)
        monkeypatch.setattr(
            sys, "meta_path", [_BlockImports(blocked), *sys.meta_path]
        )

    return _block


@pytest.fixture
def never_downloads(monkeypatch):
    """Record whether the node reached its 7.9 GB download."""
    calls: list[str] = []
    monkeypatch.setattr(
        image_to_3d.TripoSG, "_load_models", lambda self: calls.append("downloaded")
    )
    return calls


@pytest.fixture
def runnable_node(monkeypatch):
    """A TripoSG that believes it is on a CUDA box with an input image."""
    monkeypatch.setattr(image_to_3d, "_resolve_device", lambda: "cuda")
    monkeypatch.setattr(image_to_3d, "_warn_platform", lambda *a, **k: None)
    monkeypatch.setattr(image_to_3d, "_warn_vram", lambda *a, **k: None)
    monkeypatch.setattr(image_to_3d, "_report_stage", lambda *a, **k: None)
    return image_to_3d.TripoSG(id="triposg", image=ImageRef(uri="file:///tmp/in.png"))


# ---------------------------------------------------------------------------
# the defect: a legible error instead of a bare ModuleNotFoundError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "absent",
    [
        pytest.param("skimage", id="skimage-the-one-the-pod-hit"),
        pytest.param("cv2", id="cv2-the-other-half-of-the-same-block"),
        pytest.param("pymeshlab", id="pymeshlab-in-_prepare_mesh"),
    ],
)
@pytest.mark.asyncio
async def test_process_names_the_extra_instead_of_dying_on_the_import(
    absent, block_modules, never_downloads, runnable_node
):
    block_modules(absent)

    with pytest.raises(MissingDependencyError) as raised:
        await runnable_node.process(context=None)  # type: ignore[arg-type]

    message = str(raised.value)
    assert absent in message, f"{absent} is missing but the message does not say so"
    assert "[triposg] extra" in message
    # The user needs the extra, not the individual distribution names.
    assert "nodetool-huggingface[triposg]" in (raised.value.install_hint or "")


def test_vendored_triposg_is_part_of_the_wheel_not_a_users_to_install():
    """Why the guard says nothing about the triposg package itself.

    It is not on PyPI -- `pip install git+https://github.com/VAST-AI-Research/
    TripoSG.git` fails with "does not appear to be a Python project: neither
    'setup.py' nor 'pyproject.toml' found". This repo vendors it at
    src/triposg instead and ships it in the wheel, so a correct install always
    has it and telling a user to go install it would be wrong.
    """
    import tomllib

    pyproject = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    packaged = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/triposg" in packaged, (
        "triposg is no longer shipped in the wheel; if it became something the "
        "user must install, the guard needs to say so"
    )
    assert (ROOT / "src" / "triposg" / "__init__.py").is_file()
    # ...so it never appears in a message asking the user to install it.
    assert "triposg" not in EXTRA_MODULES


@pytest.mark.asyncio
async def test_every_missing_module_is_named_at_once(
    block_modules, never_downloads, runnable_node
):
    """One run should not make the user rediscover the next missing package."""
    block_modules(*EXTRA_MODULES)

    with pytest.raises(MissingDependencyError) as raised:
        await runnable_node.process(context=None)  # type: ignore[arg-type]

    message = str(raised.value)
    for name in EXTRA_MODULES:
        assert name in message, f"{name} is missing but the message does not say so"


# ---------------------------------------------------------------------------
# the placement: before the download, not after
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_refuses_before_spending_the_download(
    block_modules, never_downloads, runnable_node
):
    block_modules("skimage")

    with pytest.raises(MissingDependencyError):
        await runnable_node.process(context=None)  # type: ignore[arg-type]

    assert never_downloads == [], (
        "process() downloaded VAST-AI/TripoSG (7.9 GB) before reporting that the "
        "node cannot run"
    )


@pytest.mark.asyncio
async def test_preload_skips_the_download_when_a_dependency_is_missing(
    monkeypatch, block_modules, never_downloads
):
    block_modules("skimage")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    node = image_to_3d.TripoSG(id="triposg")
    await node.preload_model(context=None)  # type: ignore[arg-type]

    assert never_downloads == [], (
        "preload_model downloaded the model despite a missing import"
    )


# ---------------------------------------------------------------------------
# the contrast: a complete install is untouched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preload_still_downloads_on_a_complete_install(
    monkeypatch, never_downloads
):
    """The guard must not turn a working install into a silent no-op."""
    for name in EXTRA_MODULES:
        pytest.importorskip(name)

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    node = image_to_3d.TripoSG(id="triposg")
    await node.preload_model(context=None)  # type: ignore[arg-type]

    assert never_downloads == ["downloaded"]


@pytest.mark.asyncio
async def test_process_gets_past_the_guard_on_a_complete_install(
    never_downloads, runnable_node
):
    """A complete install must not raise MissingDependencyError at all."""
    for name in EXTRA_MODULES:
        pytest.importorskip(name)

    with pytest.raises(BaseException) as raised:
        await runnable_node.process(context=None)  # type: ignore[arg-type]

    assert not isinstance(raised.value, MissingDependencyError), (
        "the guard fires on an install that has every dependency"
    )
    assert never_downloads == ["downloaded"]


# ---------------------------------------------------------------------------
# the sibling contrast that made this a defect rather than a design choice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sibling", ["Hunyuan3D", "StableFast3D", "Trellis2"])
def test_siblings_already_report_their_missing_packages(sibling):
    """Why TripoSG alone was wrong: the file's convention was already set."""
    import inspect

    source = inspect.getsource(getattr(image_to_3d, sibling))
    assert "MissingDependencyError" in source
    assert "except ImportError" in source
