from __future__ import annotations

from typing import Any

from pydantic import Field

from nodetool.metadata.types import VideoRef
from nodetool.workflows.processing_context import ProcessingContext

from .text_to_video import LTX2_GEOMETRY, LTX25, LTX25Conditioned, snap_video_frames


class LTX25Extend(LTX25Conditioned):
    """
    Continues a video past its last frame, with a matching soundtrack, using LTX-2.5.
    video, audio, generation, AI, extend, continuation, video-to-video, ltx, ltx-2.5, Lightricks

    The last frames of the source condition the start of a new clip. The output
    holds only the new frames and their sound, so join it after the source with
    a video concat node.

    Use cases:
    - Make a clip longer without a visible cut
    - Continue a generated shot with a new action or line of dialogue
    - Chain several generations into one long take
    """

    video: VideoRef = Field(
        default=VideoRef(),
        description="The clip to continue.",
    )
    context_frames: int = Field(
        default=17,
        description=(
            "Frames from the end of the source that the continuation starts from. "
            "Snapped to 8n+1. More frames keep motion smoother and cost more memory."
        ),
        ge=9,
        le=65,
    )
    num_frames: int = Field(
        default=145,
        description=(
            "Frames to generate, including the context frames, which are dropped from the output."
        ),
        ge=9,
        le=257,
    )

    @classmethod
    def get_title(cls) -> str:
        return "LTX-2.5 (Extend)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["video", "prompt", "num_frames"]

    def _context_frame_count(self) -> int:
        context = snap_video_frames(
            self.context_frames, LTX2_GEOMETRY, self.get_title()
        )
        if context >= self._snapped_num_frames():
            raise ValueError(
                f"num_frames ({self._snapped_num_frames()}) must be larger than "
                f"context_frames ({context}) to leave room for new frames."
            )
        return context

    async def _conditions(self, context: ProcessingContext) -> list[Any]:
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

        if self.video.is_empty():
            raise ValueError("Connect the clip to continue.")

        count = self._context_frame_count()
        frames = await context.video_to_frames(self.video, fps=round(self.frame_rate))
        if len(frames) < count:
            raise ValueError(
                f"The source clip has {len(frames)} frames at {self.frame_rate:g} fps, "
                f"fewer than the {count} context frames. Lower context_frames."
            )
        tail = [frame.convert("RGB") for frame in frames[-count:]]
        return [LTX2VideoCondition(frames=tail, index=0, strength=1.0)]

    async def process(self, context: ProcessingContext) -> LTX25.OutputType:
        conditions = await self._conditions(context)
        return await self._generate(
            context, conditions, skip_frames=self._context_frame_count()
        )
