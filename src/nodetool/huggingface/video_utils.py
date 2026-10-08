from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from nodetool.media.video.video_utils import export_to_video
from nodetool.metadata.types import VideoRef
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.processing_offload import _in_thread


async def _mp4_bytes(encode: Callable[[str], Any]) -> bytes:
    """Run ``encode(path)`` into a closed temp file and return its bytes.

    The file is closed before encoding because Windows cannot reopen a
    ``NamedTemporaryFile`` that is still open.
    """
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        await _in_thread(encode, path)
        return await asyncio.to_thread(Path(path).read_bytes)
    finally:
        with suppress(OSError):
            os.unlink(path)


async def video_from_frames_with_audio(
    context: ProcessingContext,
    frames: list[Image.Image] | list[np.ndarray] | np.ndarray,
    audio: Any,
    audio_sample_rate: int,
    fps: int = 24,
    name: str | None = None,
    parent_id: str | None = None,
) -> VideoRef:
    """Mux generated frames and their soundtrack into a single MP4.

    Used by models that generate video and audio jointly (MiniMax-H3), where the
    two come out of the pipeline as separate outputs and muxing is left to the
    caller. ``audio`` is a ``[channels, samples]`` tensor, as diffusers'
    ``encode_video`` expects.
    """
    from diffusers.utils.export_utils import encode_video

    frame_count = len(frames)
    width, height = 0, 0

    if frame_count > 0:
        first_frame = frames[0]
        if isinstance(first_frame, Image.Image):
            width, height = first_frame.size
        elif first_frame.ndim >= 2:
            height, width = first_frame.shape[:2]

    metadata = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "format": "mp4",
        "duration_seconds": frame_count / fps if fps > 0 else None,
        "has_audio": True,
        "audio_sample_rate": audio_sample_rate,
    }

    content = await _mp4_bytes(
        lambda path: encode_video(
            frames, fps, path, audio=audio, audio_sample_rate=audio_sample_rate
        )
    )

    return await context.video_from_bytes(
        content,
        name=name,
        parent_id=parent_id,
        metadata=metadata,
    )


async def video_from_frames(
    context: ProcessingContext,
    frames: list[Image.Image] | list[np.ndarray],
    fps: int = 30,
    name: str | None = None,
    parent_id: str | None = None,
) -> VideoRef:
    frame_count = len(frames)
    width, height = 0, 0

    if frame_count > 0:
        first_frame = frames[0]
        if isinstance(first_frame, Image.Image):
            width, height = first_frame.size
        elif first_frame.ndim >= 2:
            height, width = first_frame.shape[:2]

    metadata = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "format": "mp4",
        "duration_seconds": frame_count / fps if fps > 0 else None,
    }

    content = await _mp4_bytes(lambda path: export_to_video(frames, path, fps=fps))

    return await context.video_from_bytes(
        content,
        name=name,
        parent_id=parent_id,
        metadata=metadata,
    )
