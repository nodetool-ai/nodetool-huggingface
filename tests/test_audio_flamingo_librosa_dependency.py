"""transformers decodes an audio path with librosa when torchcodec is absent.

`AudioFlamingo` with `nvidia/audio-flamingo-3-hf` failed on a RunPod 4xA40 pod
running `main` with::

    load_audio requires the librosa library. But that was not found in your
    environment. You can install them with pip: `pip install librosa`

Environment: diffusers 0.40.0, transformers 5.15.1, torchaudio 2.9.0+cu128,
pyannote.audio 3.4.0, Python 3.11.

The path is entirely inside transformers. `AudioFlamingo.process` exports the
input audio to a temp WAV and puts its **path** in the chat template::

    {"type": "audio", "path": tmp.name}

`ProcessorMixin.apply_chat_template` turns that into
``load_audio(fname, sampling_rate=..., backend=load_audio_backend)``
(`processing_utils.py`), with the backend defaulting to ``"auto"``.
``transformers.audio_utils.load_audio`` then resolves ``"auto"`` to torchcodec
when ``is_torchcodec_available()`` and the version is at least 0.3.0, and to
librosa otherwise -- ending at ``requires_backends(load_audio, ["librosa"])``,
which is the message above. The pod had neither.

`AudioFlamingo` is the only node in this package on that path. Every other
audio node decodes with ``context.audio_to_numpy`` and hands the model an
``np.ndarray``, which ``load_audio`` returns untouched on its first line --
``test_load_audio_returns_an_ndarray_untouched`` proves that against the
installed transformers rather than assuming it.

These checks read the manifest and the installed upstream source. They never
probe ``import librosa``: librosa sits in a development environment as somebody
else's transitive (here whisperlivekit, fastrtc-moonshine-onnx, and
transformers' own `audio` extra all declare it), so importability is green
locally and red only on a clean deployment. What was missing is the
*declaration*. The premise is pinned alongside it, so a transformers release
that stops offering a librosa backend makes this fail rather than silently
keeping a dependency nobody needs.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
import tomllib
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
NODES = REPO_ROOT / "src" / "nodetool" / "nodes" / "huggingface"
AUDIO_TEXT_TO_TEXT = NODES / "audio_text_to_text.py"

# Every other node in this package that consumes an AudioRef. Each one decodes
# to a numpy array itself, which is why none of them needs librosa.
ARRAY_FED_AUDIO_NODE_MODULES = (
    "audio_classification.py",
    "automatic_speech_recognition.py",
    "audio_to_audio.py",
    "speaker_diarization.py",
)


def _declared_base_dependencies() -> set[str]:
    """Distribution names in [project].dependencies, normalised and lowercased."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    names = set()
    for spec in data["project"]["dependencies"]:
        name = re.split(r"[\[<>=!~; ]", spec.strip(), maxsplit=1)[0]
        names.add(name.lower().replace("_", "-"))
    return names


def _source_of(module_name: str, *attributes: str) -> str:
    """Source of ``module.attr[.attr...]`` in the *installed* transformers."""
    target = pytest.importorskip(module_name)
    for attribute in attributes:
        target = getattr(target, attribute)
    return inspect.getsource(target)


def _requires_backends_arguments(source: str) -> list[list[str]]:
    """Every string list handed to ``requires_backends(...)`` in ``source``."""
    tree = ast.parse(textwrap.dedent(source))
    found: list[list[str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "requires_backends"
        ):
            continue
        for argument in node.args[1:]:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                found.append([argument.value])
            elif isinstance(argument, (ast.List, ast.Tuple)):
                found.append(
                    [
                        element.value
                        for element in argument.elts
                        if isinstance(element, ast.Constant)
                        and isinstance(element.value, str)
                    ]
                )
    return found


# ---------------------------------------------------------------------------
# the premise, read from the installed transformers rather than assumed
# ---------------------------------------------------------------------------


def test_load_audio_requires_librosa_when_torchcodec_is_absent():
    source = _source_of("transformers.audio_utils", "load_audio")

    backends = [name for names in _requires_backends_arguments(source) for name in names]
    assert "librosa" in backends, (
        "transformers.audio_utils.load_audio no longer has a librosa backend; "
        "if that ships in the transformers range this package depends on, the "
        "librosa dependency can go"
    )
    assert "is_torchcodec_available" in source, (
        "load_audio no longer prefers torchcodec -- re-read this file's premise "
        "before trusting the rest of it"
    )


def test_apply_chat_template_decodes_audio_paths_through_load_audio():
    """The bridge from the node's chat message to load_audio."""
    source = _source_of(
        "transformers.processing_utils", "ProcessorMixin", "apply_chat_template"
    )
    assert "load_audio(" in source
    assert '"path"' in source, (
        "apply_chat_template no longer collects audio parts by 'path', which is "
        "the key AudioFlamingo sends"
    )


def test_load_audio_returns_an_ndarray_untouched():
    """Why the other audio nodes never reach a decoding backend at all."""
    from transformers.audio_utils import load_audio

    samples = np.zeros(16, dtype=np.float32)
    assert load_audio(samples) is samples


# ---------------------------------------------------------------------------
# the declaration, and the node it exists for
# ---------------------------------------------------------------------------


def test_librosa_is_a_base_dependency():
    assert "librosa" in _declared_base_dependencies(), (
        "librosa must be a BASE dependency, not an extra: the Dockerfile "
        "installs this package with no extras, and AudioFlamingo cannot decode "
        "its own input without a load_audio backend"
    )


def test_audio_flamingo_hands_transformers_a_path():
    source = AUDIO_TEXT_TO_TEXT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert "AudioFlamingo" in classes
    assert '"type": "audio"' in source and '"path"' in source
    assert "apply_chat_template" in source


@pytest.mark.parametrize("module", ARRAY_FED_AUDIO_NODE_MODULES)
def test_other_audio_nodes_decode_to_arrays_themselves(module):
    """The contrast that keeps this a one-node problem rather than a six-node one."""
    source = (NODES / module).read_text(encoding="utf-8")
    assert "audio_to_numpy" in source, (
        f"{module} no longer decodes audio itself; if it now hands transformers "
        "a path or URL, this file's scope is wrong"
    )
    assert '"type": "audio"' not in source
