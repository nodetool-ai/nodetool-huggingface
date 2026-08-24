"""The package that ships a node must declare what that node needs at runtime.

`DocumentQuestionAnswering` shipped for months unable to run: transformers'
document-question-answering pipeline imports pytesseract itself whenever it is
handed an image without `word_boxes`, and no manifest in this package named
pytesseract.  The node failed only at execute time, on a machine that happened
not to have the binding installed.

An importability check would not have found it — pytesseract is installed on a
typical development machine as a transitive of something else, so `import
pytesseract` is green locally and red only on a clean deployment.  What was
actually missing was the *declaration*, so that is what these tests read: the
manifest, never the environment.
"""

import ast
import inspect
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
NODES_ROOT = REPO_ROOT / "src" / "nodetool" / "nodes"

# transformers guards an optional backend with `is_<name>_available()`.  Map the
# guard name onto the distribution that provides it.
GUARD_DISTRIBUTION = {
    "av": "av",
    "pytesseract": "pytesseract",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchcodec": "torchcodec",
    "vision": "pillow",
}

# Backends this package deliberately does not carry in its base dependencies.
# Adding an entry is a decision, which is the point of the list: a new optional
# backend fails the test until somebody writes down what they chose.
WAIVED_BACKENDS = {
    "vision": (
        "Pillow arrives transitively through nodetool-core and diffusers, both "
        "base dependencies, and is present in every install."
    ),
    "torchcodec": (
        "Declared in the f5-tts extra only.  The audio pipelines fall back to "
        "torchaudio when it is absent, so audio nodes still run without it."
    ),
}


def _declared_base_dependencies() -> set[str]:
    """Distribution names in [project].dependencies, normalised and lowercased."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    names = set()
    for spec in data["project"]["dependencies"]:
        name = re.split(r"[\[<>=!~; ]", spec.strip(), maxsplit=1)[0]
        names.add(name.lower().replace("_", "-"))
    return names


def _declared_pipeline_tasks() -> dict[str, list[str]]:
    """Every `pipeline_task="..."` literal in the node tree, by task."""
    tasks: dict[str, list[str]] = {}
    pattern = re.compile(r'pipeline_task\s*=\s*"([a-z0-9-]+)"')
    for path in NODES_ROOT.rglob("*.py"):
        for task in pattern.findall(path.read_text(encoding="utf-8")):
            tasks.setdefault(task, []).append(str(path.relative_to(REPO_ROOT)))
    return tasks


def _optional_backends(task: str) -> set[str]:
    """The optional backends transformers' implementation of `task` can reach."""
    from nodetool.huggingface.local_provider_utils import _normalize_pipeline_task
    from transformers.pipelines import check_task

    resolved, _, _ = check_task(_normalize_pipeline_task(task))
    from transformers.pipelines import SUPPORTED_TASKS

    module = inspect.getmodule(SUPPORTED_TASKS[resolved]["impl"])
    assert module is not None, resolved
    source = inspect.getsource(module)
    return set(re.findall(r"is_(\w+?)_available\(\)", source))


def test_node_tree_declares_pipeline_tasks():
    tasks = _declared_pipeline_tasks()
    assert len(tasks) >= 10, f"only found {len(tasks)} pipeline tasks — the scan is broken"


def test_pipeline_optional_backends_are_declared():
    """Every optional backend a pipeline we use can reach is declared or waived.

    This is the general form of the pytesseract bug: a node names a transformers
    pipeline, that pipeline imports something on its default path, and nothing in
    this package says so.
    """
    declared = _declared_base_dependencies()
    tasks = _declared_pipeline_tasks()
    assert tasks, "found no pipeline tasks at all — the scan is broken"

    seen_backends: set[str] = set()
    undeclared: dict[str, dict[str, object]] = {}

    for task, sites in sorted(tasks.items()):
        for backend in sorted(_optional_backends(task)):
            seen_backends.add(backend)
            if backend in WAIVED_BACKENDS:
                continue
            distribution = GUARD_DISTRIBUTION.get(backend)
            if distribution is None:
                undeclared[backend] = {
                    "task": task,
                    "nodes": sorted(set(sites)),
                    "problem": "unknown backend — add it to GUARD_DISTRIBUTION",
                }
            elif distribution not in declared:
                undeclared[backend] = {
                    "task": task,
                    "nodes": sorted(set(sites)),
                    "problem": f"'{distribution}' is not in [project].dependencies",
                }

    assert len(seen_backends) >= 4, (
        f"found only {sorted(seen_backends)} — transformers probably renamed its "
        "availability guards, so this check no longer reads anything"
    )
    assert not undeclared, (
        "a transformers pipeline these nodes use imports a backend this package "
        "does not declare; add it to [project].dependencies, or to "
        f"WAIVED_BACKENDS with a reason:\n{undeclared}"
    )


def test_document_qa_still_needs_pytesseract():
    """The premise behind the pytesseract dependency, checked rather than assumed.

    If transformers stops reaching for pytesseract here, this fails and the
    dependency can be dropped.
    """
    assert "pytesseract" in _optional_backends("document-question-answering")
    assert "pytesseract" in _declared_base_dependencies(), (
        "pytesseract must be a BASE dependency, not an extra: the Dockerfile "
        "installs this package with no extras, and a plain "
        "`pip install nodetool-huggingface` must be able to run doc-QA."
    )


def test_document_question_answering_node_uses_that_pipeline():
    """Ties the dependency to a node that actually ships."""
    source = (NODES_ROOT / "huggingface" / "document_question_answering.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert "DocumentQuestionAnswering" in classes
    assert 'pipeline_task="document-question-answering"' in source


@pytest.mark.parametrize("backend, reason", sorted(WAIVED_BACKENDS.items()))
def test_waivers_carry_a_reason(backend, reason):
    assert reason.strip(), backend
