"""Regression: SpeakerDiarization could not even import pyannote.audio.

On a RunPod 4xA40 pod running `main`,
``huggingface.speaker_diarization.SpeakerDiarization`` with
``pyannote/speaker-diarization-3.1`` failed with::

    module 'torchaudio' has no attribute 'AudioMetaData'

Environment: diffusers 0.40.0, transformers 5.15.1, torchaudio 2.9.0+cu128,
pyannote.audio 3.4.0, Python 3.11. Read off that machine::

    torchaudio.__version__                          2.9.0+cu128
    hasattr(torchaudio, "AudioMetaData")            False
    from torchaudio.backend.common import AudioMetaData -> ModuleNotFoundError

``pyannote/audio/core/io.py`` in 3.4.0 annotates ``get_torchaudio_info`` with
``-> torchaudio.AudioMetaData``. The module has no ``from __future__ import
annotations``, so the annotation is evaluated at import time and
``import pyannote.audio`` raises before any node code runs. 3.4.0 is the last
3.x release ("pin pyannote.{core,database,metrics,pipeline} dependencies as
future releases of these packages will break the 3.x branch"), so no 3.x
release fixes this.

pyannote.audio 4.0.0 switched audio I/O from torchaudio to torchcodec and the
symbol is gone from the package -- ``grep -rn AudioMetaData`` over the 4.0.7
wheel returns nothing. 4.0.0 also renamed ``use_auth_token`` to ``token`` and
made the diarization pipeline return a ``DiarizeOutput`` dataclass. Those two
breaking changes are what the node had to follow, and they are what the checks
below pin.

Nothing here imports pyannote.audio, torchaudio, or torchcodec: the defect is a
version contract, so the checks read the declared contract and the node's own
call shape.
"""

from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT.parent / "nodetool-core" / "src"
HF_SRC = ROOT / "src"
for _p in (str(CORE_SRC), str(HF_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nodetool.nodes.huggingface.speaker_diarization import (  # noqa: E402
    SpeakerDiarization,
)

# The last pyannote.audio release whose import breaks on torchaudio 2.9.
LAST_BROKEN_PYANNOTE = (3, 4, 0)
# 4.0.2 and 4.0.3 pin torch==2.8.0 / torchaudio==2.8.0, which contradicts this
# package's torch==2.9.0. 4.0.4 relaxed them back to >=2.8.0.
FIRST_USABLE_PYANNOTE = (4, 0, 4)


def _dependencies() -> list[str]:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["dependencies"]


def _requirement(name: str) -> str:
    for dep in _dependencies():
        if re.split(r"[<>=!~\[]", dep, maxsplit=1)[0].strip() == name:
            return dep
    raise AssertionError(f"{name} is not a declared dependency: {_dependencies()}")


def _floor(requirement: str) -> tuple[int, ...]:
    match = re.search(r">=\s*([0-9][0-9.]*)", requirement)
    assert match, f"no lower bound in {requirement!r}"
    return tuple(int(part) for part in match.group(1).split("."))


def test_pyannote_floor_excludes_every_release_that_reads_audiometadata():
    floor = _floor(_requirement("pyannote.audio"))
    assert floor > LAST_BROKEN_PYANNOTE, (
        f"pyannote.audio>={'.'.join(map(str, floor))} still allows 3.x, whose "
        "import does `from torchaudio import AudioMetaData` and dies on "
        "torchaudio 2.9"
    )
    assert floor >= FIRST_USABLE_PYANNOTE, (
        "pyannote.audio 4.0.2 and 4.0.3 pin torch==2.8.0, which contradicts "
        "this package's torch==2.9.0"
    )


def test_torchcodec_is_declared_for_pyannote_4s_audio_io():
    """pyannote.audio 4 decodes through torchcodec, which links against libtorch."""
    requirement = _requirement("torchcodec")
    assert "==" in requirement, (
        f"torchcodec must be pinned to the torch 2.9 build, got {requirement!r}"
    )


class _Pyannote4Pipeline:
    """Rejects ``use_auth_token`` exactly as pyannote.audio 4.x's signature does."""

    loaded: dict[str, Any] = {}

    @classmethod
    def from_pretrained(
        cls,
        checkpoint,
        revision=None,
        hparams_file=None,
        subfolder=None,
        token=None,
        cache_dir=None,
    ):
        cls.loaded = {"checkpoint": checkpoint, "token": token}
        return cls()

    def to(self, device):
        return self


class _FakeContext:
    device = "cpu"

    async def get_secret(self, key):
        assert key == "HF_TOKEN"
        return "hf_secret_token"


@pytest.mark.asyncio
async def test_preload_uses_the_pyannote_4_token_keyword(monkeypatch):
    """4.0.0: BREAKING(hub): rename ``use_auth_token`` to ``token``."""
    import types

    pyannote = types.ModuleType("pyannote")
    pyannote.__path__ = []  # mark as package
    audio = types.ModuleType("pyannote.audio")
    audio.Pipeline = _Pyannote4Pipeline
    pyannote.audio = audio
    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)

    node = SpeakerDiarization()
    await node.preload_model(_FakeContext())

    assert _Pyannote4Pipeline.loaded["token"] == "hf_secret_token"


@dataclass
class _Turn:
    start: float
    end: float


class _Annotation:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        assert yield_label
        for start, end, speaker in self._tracks:
            yield _Turn(start, end), None, speaker


@dataclass
class _DiarizeOutput:
    """pyannote.audio 4.x's ``SpeakerDiarization.apply`` return value."""

    speaker_diarization: _Annotation
    exclusive_speaker_diarization: _Annotation
    speaker_embeddings: Any = None


def test_to_segments_reads_the_pyannote_4_diarize_output():
    regular = _Annotation([(0.5, 2.25, "SPEAKER_00"), (2.25, 4.0, "SPEAKER_01")])
    # Exclusive diarization drops overlapping turns; picking it would be wrong.
    exclusive = _Annotation([(0.5, 2.0, "SPEAKER_00")])

    result = SpeakerDiarization._to_segments(
        _DiarizeOutput(speaker_diarization=regular, exclusive_speaker_diarization=exclusive)
    )

    assert [c.timestamp for c in result["segments"]] == [(0.5, 2.25), (2.25, 4.0)]
    assert result["speakers"] == ["SPEAKER_00", "SPEAKER_01"]


def test_to_segments_still_reads_a_bare_annotation():
    """A pipeline built with ``legacy=True`` returns the 3.x shape."""
    result = SpeakerDiarization._to_segments(_Annotation([(0.0, 1.0, "SPEAKER_00")]))
    assert result["speakers"] == ["SPEAKER_00"]
