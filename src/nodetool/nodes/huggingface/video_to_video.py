from __future__ import annotations

from enum import Enum
from typing import Any, NamedTuple

from pydantic import Field

from nodetool.integrations.huggingface.huggingface_models import HF_FAST_CACHE
from nodetool.metadata.types import HFTextToVideo, HuggingFaceModel, VideoRef
from nodetool.workflows.processing_context import ProcessingContext

from .text_to_video import (
    _DIFFUSERS_REPO_ALLOW_PATTERNS,
    LTX2_GEOMETRY,
    LTX25,
    LTX25Conditioned,
    snap_video_frames,
)


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


LTX23_DISTILLED_REPO_ID = "diffusers/LTX-2.3-Distilled-Diffusers"


class ICLoRA(NamedTuple):
    repo_id: str
    path: str
    # The reference is encoded at 1/downscale of the output resolution.
    reference_downscale: int
    # Trigger text the adapter was trained with, put in front of the prompt.
    trigger: str = ""


class LTXControl(str, Enum):
    MOTION_CONTROL = "motion_control"
    COLORIZE = "colorize"


LTX23_IC_LORAS: dict[LTXControl, ICLoRA] = {
    LTXControl.MOTION_CONTROL: ICLoRA(
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        path="ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        reference_downscale=2,
    ),
    LTXControl.COLORIZE: ICLoRA(
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Colorization",
        path="ltx-2.3-22b-ic-lora-colorization-0.9.safetensors",
        reference_downscale=1,
        trigger="COLORIZE",
    ),
}


class LTXVideoControl(LTX25):
    """
    Re-renders a video under LTX-2.3 IC-LoRA control: copy its motion and camera, or colorize it.
    video, audio, generation, AI, video-to-video, motion control, pose, depth, canny, colorize, ic-lora, ltx, Lightricks

    Motion control runs Lightricks' Union-Control adapter: the output keeps the
    clip's motion, camera move, composition and pose, and the prompt paints a
    new subject and scene onto it. It reads raw video as well as canny, depth
    or pose sequences. Colorize runs the Colorization adapter on black-and-white
    footage. The output matches the clip's aspect ratio and, up to num_frames,
    its length.

    Use cases:
    - Restage a shot with a new character or setting, keeping the camera move
    - Drive a generation from a pose, depth or edge sequence
    - Colorize archival or black-and-white footage
    """

    model: HFTextToVideo = Field(
        default=HFTextToVideo(repo_id=LTX23_DISTILLED_REPO_ID),
        description="The distilled LTX-2.3 Diffusers checkpoint the IC-LoRAs were trained on.",
    )
    control: LTXControl = Field(
        default=LTXControl.MOTION_CONTROL,
        description="Motion control copies the clip's motion and camera. Colorize adds color to black-and-white footage.",
    )
    video: VideoRef = Field(
        default=VideoRef(),
        description="The clip that drives the output.",
    )
    prompt: str = Field(
        default="A monk in orange robes on a white salt flat at dawn",
        description="The new subject and scene. Leave the camera move out: the clip sets it.",
    )
    num_frames: int = Field(
        default=121,
        description="Most frames to generate. A shorter clip sets a shorter output.",
        ge=9,
        le=257,
    )
    width: int = Field(
        default=0,
        description="Output width. 0 keeps the clip's aspect ratio at 768 pixels on the long side.",
        ge=0,
        le=1280,
    )
    height: int = Field(
        default=0,
        description="Output height. 0 keeps the clip's aspect ratio.",
        ge=0,
        le=1280,
    )
    lora_strength: float = Field(
        default=1.0,
        description="Strength of the IC-LoRA adapter.",
        ge=0.0,
        le=2.0,
    )
    reference_strength: float = Field(
        default=1.0,
        description="How clean the clip looks to the model. Lower values follow it more loosely.",
        ge=0.0,
        le=1.0,
    )

    @classmethod
    def get_title(cls) -> str:
        return "LTX-2.3 Video Control (IC-LoRA)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["video", "control", "prompt"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToVideo(
                repo_id=LTX23_DISTILLED_REPO_ID,
                allow_patterns=_DIFFUSERS_REPO_ALLOW_PATTERNS,
            ),
            *(
                HFTextToVideo(repo_id=lora.repo_id, path=lora.path)
                for lora in LTX23_IC_LORAS.values()
            ),
        ]

    def get_model_id(self) -> str:
        return self.model.repo_id or LTX23_DISTILLED_REPO_ID

    def model_cache_suffix(self) -> str:
        lora = LTX23_IC_LORAS[self.control]
        return f"_iclora_{self.control.value}_{self.lora_strength:g}_{lora.path}"

    # Set by process() from the clip, which decides the output geometry.
    _clip_geometry: tuple[int, int, int] | None = None

    def _geometry(self) -> tuple[int, int, int]:
        if self._clip_geometry is None:
            raise ValueError("The output geometry comes from the clip.")
        return self._clip_geometry

    def _pipeline_class(self) -> Any:
        from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import (
            LTX2InContextPipeline,
        )

        return LTX2InContextPipeline

    async def _bind_adapters(self, pipeline: Any) -> None:
        # Each control has its own cache entry, so a cached pipeline already
        # carries its adapter and only a fresh one needs it bound.
        if getattr(pipeline, "_nodetool_ic_lora", None) == self.model_cache_suffix():
            return
        lora = LTX23_IC_LORAS[self.control]
        path = await HF_FAST_CACHE.resolve(lora.repo_id, lora.path)
        if path is None:
            raise ValueError(
                f"Download {lora.repo_id}/{lora.path} from the recommended models first."
            )
        pipeline.load_lora_weights(path, adapter_name="ic_lora")
        pipeline.set_adapters(["ic_lora"], adapter_weights=[self.lora_strength])
        pipeline._nodetool_ic_lora = self.model_cache_suffix()

    def _prompt(self) -> str:
        trigger = LTX23_IC_LORAS[self.control].trigger
        if trigger and not self.prompt.lstrip().upper().startswith(trigger):
            return f"{trigger} {self.prompt}".strip()
        return self.prompt

    def _output_geometry(
        self, clip_size: tuple[int, int], clip_frames: int
    ) -> tuple[int, int, int]:
        """(width, height, num_frames) for the output, on the lattice the reference needs."""
        multiple = (
            LTX2_GEOMETRY.dim_multiple
            * LTX23_IC_LORAS[self.control].reference_downscale
        )
        if self.width and self.height:
            width, height = self.width, self.height
        else:
            clip_width, clip_height = clip_size
            scale = 768 / max(clip_width, clip_height)
            width, height = clip_width * scale, clip_height * scale

        def snap(value: float) -> int:
            snapped = max(1, round(value / multiple)) * multiple
            return min(snapped, LTX2_GEOMETRY.max_dim // multiple * multiple)

        frames = min(self.num_frames, clip_frames)
        if frames < LTX2_GEOMETRY.min_frames:
            raise ValueError(
                f"The clip has {clip_frames} frames at {self.frame_rate:g} fps; "
                f"video control needs at least {LTX2_GEOMETRY.min_frames}."
            )
        # Round down onto 8n+1 so the output never outruns the clip.
        frames = (frames - 1) // LTX2_GEOMETRY.frame_step * LTX2_GEOMETRY.frame_step + 1
        return snap(width), snap(height), frames

    async def process(self, context: ProcessingContext) -> LTX25.OutputType:
        from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import (
            LTX2ReferenceCondition,
        )

        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")
        if self.video.is_empty():
            raise ValueError("Connect the clip that drives the output.")

        frames = await context.video_to_frames(self.video, fps=round(self.frame_rate))
        if not frames:
            raise ValueError("The clip has no frames.")
        self._clip_geometry = self._output_geometry(frames[0].size, len(frames))
        num_frames = self._clip_geometry[2]

        kwargs = self._build_pipeline_kwargs(
            self._generator(), self._progress_callback(context)
        )
        kwargs.update(
            prompt=self._prompt(),
            reference_conditions=[
                LTX2ReferenceCondition(
                    frames=[frame.convert("RGB") for frame in frames[:num_frames]],
                    strength=self.reference_strength,
                )
            ],
            reference_downscale_factor=LTX23_IC_LORAS[self.control].reference_downscale,
        )
        output = await self.run_pipeline_in_thread(**kwargs)
        return await self._video_and_audio(context, output)
