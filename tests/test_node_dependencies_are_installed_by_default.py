"""The 3D and speech nodes must work on a plain `pip install`, not on an extra.

Four nodes shipped in every install but failed on an import, because the
packages they need sat in extras almost nobody passes:

    TripoSG         skimage (measure.label, morphology.remove_small_objects)
                    pymeshlab (_prepare_mesh)
    Hunyuan3D       hy3dgen, pymeshlab
    SupertonicTTS   supertonic
    F5TTS           f5_tts

That is what `No module named 'skimage'` on a RunPod pod was: the declaration
was "correct" in that the package appeared in `[triposg]`, and useless in that
nobody installs an extra to make a shipped node run.

These checks read the manifest, never `import skimage`. Every one of these
packages sits in a development environment as somebody else's transitive
(hy3dgen alone pulls opencv-python, rembg and onnxruntime), so importability is
green locally and says nothing about a clean install. What was missing is the
declaration.

A declaration test proves declaration. It does not prove that a clean install
succeeds on every platform, and nothing here claims that -- see the PR body for
the resolution that was run separately.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
NODES = ROOT / "src" / "nodetool" / "nodes" / "huggingface"

# distribution -> the node that stops working without it
PROMOTED = {
    "scikit-image": "TripoSG",
    "pymeshlab": "TripoSG and Hunyuan3D",
    "hy3dgen": "Hunyuan3D",
    "supertonic": "SupertonicTTS",
    "f5-tts": "F5TTS",
}

# Deliberately NOT base dependencies. Each entry is (distribution, why).
WITHHELD = {
    "diso": "sdist-only, compiles against a CUDA toolchain; TripoSG guards it",
    "rembg": "only StableFast3D imports it, and sf3d is git-only",
    "paddlepaddle": "195 MB wheel, platform-fragile; stays in the ocr extra",
    "paddleocr": "stays in the ocr extra with paddlepaddle",
    "pyopenjtalk": "sdist-only, compiles; stays in kokoro-ja",
    "opencv-python": "cv2 already resolves via opencv-python-headless",
}


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _normalise(spec: str) -> str:
    return re.split(r"[\[<>=!~; ]", spec.strip(), maxsplit=1)[0].lower().replace("_", "-")


def _base_dependencies() -> set[str]:
    return {_normalise(spec) for spec in _project()["dependencies"]}


@pytest.mark.parametrize(
    "distribution,node", sorted(PROMOTED.items()), ids=sorted(PROMOTED)
)
def test_promoted_package_is_a_base_dependency(distribution, node):
    assert distribution in _base_dependencies(), (
        f"{distribution} must be a BASE dependency: {node} ships in every "
        "install, the Dockerfile installs this package with no extras, and the "
        "node fails on an import without it"
    )


@pytest.mark.parametrize(
    "distribution,reason", sorted(WITHHELD.items()), ids=sorted(WITHHELD)
)
def test_withheld_package_stays_out_of_base(distribution, reason):
    """Promoting these would be wrong; the reason travels with the check."""
    assert distribution not in _base_dependencies(), (
        f"{distribution} must NOT be a base dependency: {reason}"
    )


def test_cv2_is_reachable_without_declaring_opencv_python():
    """TripoSG imports cv2, and mistral-common already supplies it.

    `mistral-common[image]` pulls opencv-python-headless, which provides the
    same `cv2` module. Every cv2 call in this package is core imgproc, so the
    headless build serves all of them -- and declaring opencv-python as well
    would put two providers of one module in the environment.
    """
    specs = {_normalise(s): s for s in _project()["dependencies"]}
    assert "mistral-common" in specs
    assert "[image]" in specs["mistral-common"], (
        "mistral-common no longer requests its image extra, which is what "
        "brings opencv-python-headless -- cv2 may no longer resolve"
    )

    used = set()
    for path in NODES.glob("*.py"):
        used.update(re.findall(r"\bcv2\.([A-Za-z_0-9]+)", path.read_text("utf-8")))
    assert used, "no cv2 usage found; this check would pass vacuously"

    gui_only = {"imshow", "waitKey", "namedWindow", "destroyAllWindows", "createTrackbar"}
    assert not (used & gui_only), (
        f"these nodes now call GUI-only cv2 functions {sorted(used & gui_only)}, "
        "which opencv-python-headless does not provide"
    )


def test_emptied_extras_survive_as_aliases():
    """`pip install nodetool-huggingface[hunyuan3d]` must not become an error."""
    extras = _project()["optional-dependencies"]
    for name in ("supertonic", "f5-tts", "hunyuan3d"):
        assert name in extras, (
            f"the {name} extra was removed rather than emptied; it is still "
            "named in scripts, Dockerfiles and this package's node docstrings"
        )
        assert extras[name] == [], f"{name} should be empty now that it is base"


def test_triposg_extra_still_carries_what_stayed_optional():
    extras = _project()["optional-dependencies"]
    triposg = {_normalise(s) for s in extras["triposg"]}
    assert triposg == {"diso", "rembg"}, (
        "the triposg extra should hold exactly the two packages that did not "
        f"get promoted, got {sorted(triposg)}"
    )


def test_meta_extras_still_resolve_to_declared_names():
    """all-3d / all-3d-pypi reference other extras; those must still exist."""
    extras = _project()["optional-dependencies"]
    for meta in ("all-3d-pypi", "all-3d"):
        for spec in extras[meta]:
            named = re.search(r"\[([^\]]+)\]", spec)
            assert named, f"{meta} entry {spec!r} names no extra"
            for referenced in named.group(1).split(","):
                assert referenced.strip() in extras, (
                    f"{meta} references a {referenced.strip()} extra that no "
                    "longer exists"
                )


def test_the_git_only_nodes_are_still_honestly_unavailable():
    """This PR does not make sf3d/tsr/trellis2/o_voxel installable, and must not pretend to."""
    base = _base_dependencies()
    for distribution in ("sf3d", "tsr", "trellis2", "o-voxel", "triposg"):
        assert distribution not in base, (
            f"{distribution} cannot be a dependency: it has no PyPI release "
            "(triposg is vendored at src/triposg instead)"
        )
