from __future__ import annotations

import asyncio
import tempfile

import numpy as np
from PIL import Image

from nodetool.media.video.video_utils import export_to_video
from nodetool.metadata.types import VideoRef
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.processing_offload import _in_thread


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

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp:
        await _in_thread(export_to_video, frames, temp.name, fps=fps)
        temp.seek(0)
        content = await asyncio.to_thread(temp.read)

    return await context.video_from_bytes(
        content,
        name=name,
        parent_id=parent_id,
        metadata=metadata,
    )
