"""A third-party import must be declared or guarded, so it cannot ship broken.

`SplitSentences` imported `langchain_text_splitters` and `langchain_core`
straight inside `gen_process`, with no `try`/`except` and no entry anywhere in
`pyproject.toml`.  The node failed on any clean install with `No module named
'langchain_text_splitters'` — proven on a real worker, not merely inferred
from reading the code.  `VisualizeObjectDetection`'s bare `import matplotlib`
had the same shape.

Both were fixed a second time by removing the import instead of declaring it:
`SplitSentences` now chunks with `transformers`' own tokenizer (already a
base dependency) instead of LangChain's wrapper around it, and
`VisualizeObjectDetection` now draws with `PIL.ImageDraw` (already a base
dependency, and already used in the same method) instead of matplotlib.
Neither module is imported or declared anywhere in this package any more —
see `test_matplotlib_and_langchain_are_fully_removed` below.

This is the general form of the bug `tests/test_pipeline_backend_dependencies.py`
fixed for pytesseract: a module imported unconditionally at module or function
scope, with no `try`/`except ImportError` anywhere in the file to catch its
absence, and no declaration in `pyproject.toml` (base `dependencies` or any
`optional-dependencies` extra) to guarantee — or admit the possibility of
choosing — its presence.

This test reads the manifest and the AST, never the environment: a module
already sitting in the local site-packages as somebody else's transitive proves
nothing about a clean install.
"""

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
SRC_ROOT = REPO_ROOT / "src" / "nodetool"

# Vendored source trees that ship inside this package's own wheel
# ([tool.hatch.build.targets.wheel] packages).  Importing them is not a
# dependency at all — they are this distribution's own code, not something
# `pip` resolves separately.
VENDORED_PACKAGES = {"triposg", "RealESRGAN"}

# Import name -> PyPI distribution name, for the cases where they differ.
IMPORT_TO_DISTRIBUTION = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
}

# Modules this scan would otherwise flag, kept out on a documented, specific
# basis.  Every entry is a decision, which is the point of the list: a new
# unguarded-and-undeclared import fails the test until somebody writes down
# why it is safe.
EXPLICITLY_EXEMPT = {
    "numpy": (
        "Required unconditionally by torch, diffusers, transformers and "
        "accelerate (all base dependencies) — always present."
    ),
    "PIL": (
        "diffusers declares plain 'Pillow' (no extra marker) as a base "
        "requirement — always present."
    ),
    "pydantic": (
        "Required unconditionally by mistral-common (a base dependency) — "
        "always present."
    ),
    "httpx": (
        "huggingface_hub declares 'httpx<1,>=0.23.0' unconditionally (a base "
        "dependency) — always present."
    ),
    "imageio": (
        "image_text_to_text.extract_video_frames only imports imageio after "
        "nodetool.media.video.video_utils._is_imageio_available() (an "
        "importlib.util.find_spec probe, not a try/except) confirms it is "
        "present, and falls back to OpenCV with an actionable error "
        "otherwise. Guarded, just not with a try/except at the import site."
    ),
    "nunchaku": (
        "The SVDQuant nunchaku runtime has no PyPI extra in this package — "
        "it is a manual/BYO install like the git-only 3D backends "
        "(requirements/nunchaku.txt), not something 'pip install "
        "nodetool-huggingface[...]' can resolve today. nunchaku_pipelines.py "
        "probes availability via is_nunchaku_available() before use; the "
        "call sites this scan flags in image_to_image.py are reached only "
        "after that probe passes. Pre-existing gap, tracked separately — "
        "out of scope here (see nodetool-huggingface#65: the docker/extras "
        "question is the maintainer's to answer)."
    ),
    "cv2": (
        "TripoSG._prepare_image uses cv2 unguarded, gated only by the "
        "class's own runtime_availability()/preload_model contract, not a "
        "try/except at the import site. Declared under the [triposg] extra "
        "(this fix); tightening the guard to match sf3d/tsr/trellis2's "
        "try/except-in-preload_model pattern is a separate, larger change to "
        "an already-reviewed node — out of scope here."
    ),
    "pymeshlab": (
        "Used unguarded in TripoSG._simplify_mesh and Hunyuan3D's pipeline, "
        "gated only by runtime_availability()/preload_model, not a "
        "try/except at the import site. Declared under both the [triposg] "
        "and [hunyuan3d] extras. Same pre-existing pattern as cv2 above — "
        "out of scope here."
    ),
    "skimage": (
        "Used unguarded in TripoSG's mesh post-processing, gated only by "
        "runtime_availability()/preload_model. Declared under the [triposg] "
        "extra as scikit-image. Same pre-existing pattern as cv2 above — out "
        "of scope here."
    ),
}


def _declared_names() -> tuple[set[str], set[str]]:
    """(base dependency names, names declared in any extra) — normalised."""

    def _norm(spec: str) -> str:
        name = re.split(r"[\[<>=!~; ]", spec.strip(), maxsplit=1)[0]
        return name.lower().replace("_", "-")

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    base = {_norm(spec) for spec in data["project"]["dependencies"]}
    extra_names: set[str] = set()
    for specs in data["project"].get("optional-dependencies", {}).values():
        for spec in specs:
            name = _norm(spec)
            if name.startswith("nodetool-huggingface"):
                continue
            extra_names.add(name)
    return base, extra_names


def _is_guarded_import(node: ast.AST, try_blocks: list[ast.Try]) -> bool:
    """Is `node` (an Import/ImportFrom) inside a try/except that catches
    ImportError/ModuleNotFoundError (or a bare/broad except)?"""
    for try_node in try_blocks:
        if node not in ast.walk(try_node):
            continue
        for handler in try_node.handlers:
            if handler.type is None:
                return True
            names = (
                [n.id for n in handler.type.elts if isinstance(n, ast.Name)]
                if isinstance(handler.type, ast.Tuple)
                else (
                    [handler.type.id] if isinstance(handler.type, ast.Name) else []
                )
            )
            if any(n in ("ImportError", "ModuleNotFoundError", "Exception") for n in names):
                return True
    return False


def _unguarded_third_party_imports() -> dict[str, set[str]]:
    """third-party top-level module name -> files importing it with no guard."""
    stdlib = set(sys.stdlib_module_names)
    findings: dict[str, set[str]] = {}

    for path in sorted(SRC_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        try_blocks = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]

        # A module is "guarded" for the whole file if this file catches its
        # absence anywhere — mirroring this codebase's own convention of a
        # small try/except probe (often in preload_model) followed by a plain
        # import in the loader it gates (e.g. StableFast3D/TripoSR/Trellis2's
        # preload_model vs. _load_model/_load_pipeline).
        guarded_in_file: set[str] = set()
        for try_node in try_blocks:
            for n in ast.walk(try_node):
                if isinstance(n, ast.Import):
                    mods = [a.name.split(".")[0] for a in n.names]
                elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                    mods = [n.module.split(".")[0]]
                else:
                    continue
                if _is_guarded_import(n, try_blocks):
                    guarded_in_file.update(mods)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level or node.module is None:
                    continue
                mods = [node.module.split(".")[0]]
            else:
                continue

            for top in mods:
                if top == "nodetool" or top in stdlib or top in VENDORED_PACKAGES:
                    continue
                if top in guarded_in_file:
                    continue
                findings.setdefault(top, set()).add(str(path.relative_to(REPO_ROOT)))

    return findings


def test_scan_finds_a_nontrivial_number_of_third_party_imports():
    """Proves the AST walk actually inspects the tree instead of matching
    nothing — a scan that silently skips every file would also report zero
    problems."""
    findings = _unguarded_third_party_imports()
    assert len(findings) >= 15, f"only found {len(findings)} modules — the scan is broken"


def test_unguarded_imports_are_declared_or_exempt():
    findings = _unguarded_third_party_imports()
    base, extra_names = _declared_names()
    declared = base | extra_names

    unresolved = {}
    for module, files in sorted(findings.items()):
        if module in EXPLICITLY_EXEMPT:
            continue
        dist = IMPORT_TO_DISTRIBUTION.get(module, module).lower().replace("_", "-")
        if dist not in declared:
            unresolved[module] = files

    assert not unresolved, (
        "these third-party imports are neither guarded (try/except "
        "ImportError) nor declared anywhere in pyproject.toml — they will "
        "raise ModuleNotFoundError on a clean install:\n"
        + "\n".join(
            f"  {module}: {sorted(files)}" for module, files in unresolved.items()
        )
        + "\n\nDeclare the module in [project.dependencies] or an "
        "optional-dependencies extra (guarding the import with a helpful "
        "error naming the extra), or add it to EXPLICITLY_EXEMPT above with "
        "a reason."
    )


@pytest.mark.parametrize(
    "module", ["langchain_text_splitters", "langchain_core", "matplotlib"]
)
def test_matplotlib_and_langchain_are_fully_removed(module):
    """Pin the regression this test was originally written for, in its
    current form: `SplitSentences` and `VisualizeObjectDetection` were
    rewritten to avoid langchain/matplotlib entirely, so neither module
    should be imported anywhere in this package, nor declared in
    pyproject.toml (base dependencies or any extra) — a revert of either
    rewrite, or a stray re-import, fails this loudly and by name."""
    findings = _unguarded_third_party_imports()
    assert module not in findings, (
        f"{module} is imported (unguarded) in {sorted(findings[module])} — "
        "it was supposed to be removed, not re-declared"
    )

    base, extra_names = _declared_names()
    dist = module.lower().replace("_", "-")
    assert dist not in (base | extra_names), (
        f"{module} is still declared in pyproject.toml — it was supposed to "
        "be removed by rewriting the one node that needed it"
    )
