"""Shared validation and temporary-file helpers for local TTS adapters."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator

import numpy as np


@contextmanager
def temporary_reference_audio(data: bytes, suffix: str = ".wav") -> Iterator[str]:
    """Write reference audio for path-only upstream APIs and always remove it."""
    if not data:
        raise ValueError("Reference audio is required")
    with NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
        temporary.write(data)
        path = temporary.name
    try:
        yield path
    finally:
        Path(path).unlink(missing_ok=True)


def normalized_mono_audio(value: object, model_name: str) -> np.ndarray:
    """Return finite mono float32 audio, rejecting malformed model output."""
    audio = np.asarray(value, dtype=np.float32).squeeze()
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"{model_name} returned empty or non-mono audio")
    if not np.isfinite(audio).all():
        raise ValueError(f"{model_name} returned non-finite audio")
    return np.clip(audio, -1.0, 1.0)


def raise_if_cancelled(context: object, model_name: str) -> None:
    """Raise cancellation consistently for contexts exposing either API form."""
    cancelled = getattr(context, "is_cancelled", False)
    if bool(cancelled() if callable(cancelled) else cancelled):
        import asyncio

        raise asyncio.CancelledError(f"{model_name} synthesis cancelled")
