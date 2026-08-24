"""diffusers' Wan image-to-video pipeline cleans every prompt with ftfy, unguarded.

`Wan_FLF2V` failed on a RunPod A40 with `name 'ftfy' is not defined` while
`Wan_T2V` passed in the same session, on the same pod, against the same
diffusers install.  The difference is in diffusers, not in this package:
`pipeline_wan.py` guards the *call* (`if is_ftfy_available(): text =
ftfy.fix_text(text)`), and its sibling `pipeline_wan_i2v.py` guards only the
import and then calls `ftfy.fix_text(text)` unconditionally.  Both files import
ftfy behind `if is_ftfy_available()`, so without the distribution installed the
name is simply never bound and the i2v file raises `NameError` on the first
prompt it is handed.

`Wan_I2V` and `Wan_FLF2V` both run that pipeline, so neither node can process a
single prompt on an install that lacks ftfy.

This test reads the manifest and the installed diffusers source, never a
`import ftfy` probe: ftfy sits in a typical development environment as somebody
else's transitive, so importability is green locally and red only on a clean
deployment.  What was missing is the *declaration*, so that is what is checked —
along with the premise behind it, so that a diffusers release which guards the
call makes this fail rather than silently keeping a dependency nobody needs.
"""

import ast
import inspect
import re
import textwrap
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
IMAGE_TO_VIDEO = REPO_ROOT / "src" / "nodetool" / "nodes" / "huggingface" / "image_to_video.py"


def _declared_base_dependencies() -> set[str]:
    """Distribution names in [project].dependencies, normalised and lowercased."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    names = set()
    for spec in data["project"]["dependencies"]:
        name = re.split(r"[\[<>=!~; ]", spec.strip(), maxsplit=1)[0]
        names.add(name.lower().replace("_", "-"))
    return names


def _basic_clean_source(module_name: str) -> str:
    """The source of `basic_clean` in one diffusers Wan pipeline module."""
    module = pytest.importorskip(module_name)
    return inspect.getsource(module.basic_clean)


def _calls_ftfy_unguarded(source: str) -> bool:
    """True when `ftfy.fix_text(...)` runs with no `is_ftfy_available()` test above it."""
    tree = ast.parse(textwrap.dedent(source))

    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.dump(node.test)
        if "is_ftfy_available" not in test:
            continue
        for child in ast.walk(node):
            guarded.add(id(child))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fix_text"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ftfy"
            and id(node) not in guarded
        ):
            return True
    return False


def test_wan_i2v_pipeline_calls_ftfy_unguarded():
    """The premise, read from the installed diffusers rather than assumed."""
    i2v = _basic_clean_source("diffusers.pipelines.wan.pipeline_wan_i2v")
    assert "ftfy.fix_text" in i2v, (
        "diffusers no longer cleans Wan i2v prompts with ftfy — re-read this "
        "file's premise before trusting the rest of it"
    )
    assert _calls_ftfy_unguarded(i2v), (
        "pipeline_wan_i2v.basic_clean now guards its ftfy call; if that ships in "
        "the diffusers range this package depends on, the ftfy dependency can go"
    )


def test_wan_t2v_pipeline_guards_its_ftfy_call():
    """The contrast that explains why Wan_T2V passed on the same pod."""
    t2v = _basic_clean_source("diffusers.pipelines.wan.pipeline_wan")
    assert "ftfy.fix_text" in t2v
    assert not _calls_ftfy_unguarded(t2v), (
        "pipeline_wan.basic_clean is expected to guard its ftfy call — the "
        "guarded/unguarded split between the sibling files is the whole reason "
        "Wan_T2V ran without ftfy and Wan_I2V could not"
    )


def test_ftfy_is_a_base_dependency():
    assert "ftfy" in _declared_base_dependencies(), (
        "ftfy must be a BASE dependency, not an extra: the Dockerfile installs "
        "this package with no extras, and Wan_I2V / Wan_FLF2V raise NameError on "
        "their first prompt without it"
    )


def test_wan_image_to_video_nodes_use_that_pipeline():
    """Ties the dependency to nodes that actually ship."""
    source = IMAGE_TO_VIDEO.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert {"Wan_I2V", "Wan_FLF2V"} <= classes
    assert "WanImageToVideoPipeline" in source
