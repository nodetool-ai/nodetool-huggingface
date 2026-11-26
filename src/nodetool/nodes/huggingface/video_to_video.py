from __future__ import annotations

import math
from enum import Enum
from typing import Any, NamedTuple

from pydantic import Field

from PIL import Image

from nodetool.metadata.types import HFTextToVideo, HuggingFaceModel, ImageRef, VideoRef
from nodetool.workflows.processing_context import ProcessingContext

from .text_to_video import (
    _DIFFUSERS_REPO_ALLOW_PATTERNS,
    LTX2_GEOMETRY,
    LTX25,
    LTX25Conditioned,
    LoraAdapter,
    snap_video_frames,
    source_soundtrack,
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
LTX25_REPO_ID = "Lightricks/LTX-2.5-Diffusers"


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

# Community LoRA, mirrored in a repo of about 700 GB: download only this file.
LTX23_INGREDIENTS_LORA = ICLoRA(
    repo_id="DeepBeepMeep/LTX-2",
    path="ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
    reference_downscale=1,
)
# Gated: accept the license on Hugging Face before downloading.
LTX25_UPSCALE_LORA = ICLoRA(
    repo_id="Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler",
    path="ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
    reference_downscale=2,
)

# Upscale outputs up to 1536 pixels on the long side: twice a 768x448 LTX draft.
LTX_UPSCALE_MAX_DIM = 1536
INGREDIENTS_SHEET_SIZE = (1536, 896)
INGREDIENTS_GUTTER = 16
INGREDIENTS_MAX_IMAGES = 8


def compose_ingredients_sheet(images: list[Image.Image]) -> Image.Image:
    """Tile images on a black 1536x896 sheet, each fitted into its cell.

    One image is returned unchanged. This is the sheet layout Phosphene feeds
    the ingredients LoRA.
    """
    if len(images) == 1:
        return images[0].convert("RGB")
    sheet_w, sheet_h = INGREDIENTS_SHEET_SIZE
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    cell_w = (sheet_w - INGREDIENTS_GUTTER * (cols + 1)) // cols
    cell_h = (sheet_h - INGREDIENTS_GUTTER * (rows + 1)) // rows
    sheet = Image.new("RGB", (sheet_w, sheet_h), (0, 0, 0))
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        fitted = image.convert("RGB")
        scale = min(cell_w / fitted.width, cell_h / fitted.height)
        fitted = fitted.resize(
            (max(1, round(fitted.width * scale)), max(1, round(fitted.height * scale))),
            Image.Resampling.LANCZOS,
        )
        left = INGREDIENTS_GUTTER + col * (cell_w + INGREDIENTS_GUTTER)
        top = INGREDIENTS_GUTTER + row * (cell_h + INGREDIENTS_GUTTER)
        sheet.paste(
            fitted,
            (left + (cell_w - fitted.width) // 2, top + (cell_h - fitted.height) // 2),
        )
    return sheet


def ingredients_prompt(sheet: str, action: str) -> str:
    """The prompt format the ingredients LoRA was trained on."""
    sheet, action = sheet.strip(), action.strip()
    if sheet and action:
        return f"Reference sheet: {sheet}\n\nGenerated video: {action}"
    if action:
        return f"Generated video: {action}"
    return sheet or "the subjects from the reference sheet, cinematic"


class _LTXInContext(LTX25):
    """LTX-2 with an IC-LoRA: a reference clip is encoded next to the latents.

    Runs Diffusers' ``LTX2InContextPipeline``. Subclasses pick the adapter and
    build the reference frames.
    """

    lora_strength: float = Field(
        default=1.0,
        description="Strength of the IC-LoRA adapter.",
        ge=0.0,
        le=2.0,
    )
    reference_strength: float = Field(
        default=1.0,
        description="How clean the reference looks to the model. Lower values follow it more loosely.",
        ge=0.0,
        le=1.0,
    )

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not _LTXInContext

    def _ic_lora(self) -> ICLoRA:
        raise NotImplementedError

    def _adapters(self) -> list[LoraAdapter]:
        lora = self._ic_lora()
        return [
            LoraAdapter("ic_lora", lora.repo_id, lora.path, self.lora_strength),
            *super()._adapters(),
        ]

    # Set by process() before the run, from the reference.
    _clip_geometry: tuple[int, int, int] | None = None

    def _geometry(self) -> tuple[int, int, int]:
        if self._clip_geometry is None:
            raise ValueError("The output geometry comes from the reference.")
        return self._clip_geometry

    def _pipeline_class(self) -> Any:
        from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import (
            LTX2InContextPipeline,
        )

        return LTX2InContextPipeline

    def _prompt(self) -> str:
        trigger = self._ic_lora().trigger
        if trigger and not self.prompt.lstrip().upper().startswith(trigger):
            return f"{trigger} {self.prompt}".strip()
        return self.prompt

    async def _run_reference(
        self,
        context: ProcessingContext,
        frames: list[Image.Image],
        conditioning_attention_strength: float = 1.0,
    ) -> Any:
        """Run the pipeline with ``frames`` as the reference clip; return its output."""
        from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import (
            LTX2ReferenceCondition,
        )

        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")
        kwargs = self._build_pipeline_kwargs(
            self._generator(), self._progress_callback(context)
        )
        kwargs.update(
            prompt=self._prompt(),
            reference_conditions=[
                LTX2ReferenceCondition(
                    frames=[frame.convert("RGB") for frame in frames],
                    strength=self.reference_strength,
                )
            ],
            reference_downscale_factor=self._ic_lora().reference_downscale,
        )
        if conditioning_attention_strength < 1.0:
            kwargs["conditioning_attention_strength"] = conditioning_attention_strength
        return await self.run_pipeline_in_thread(**kwargs)


class LTXVideoControl(_LTXInContext):
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

    def _ic_lora(self) -> ICLoRA:
        return LTX23_IC_LORAS[self.control]

    def _output_geometry(
        self, clip_size: tuple[int, int], clip_frames: int
    ) -> tuple[int, int, int]:
        """(width, height, num_frames) for the output, on the lattice the reference needs."""
        multiple = LTX2_GEOMETRY.dim_multiple * self._ic_lora().reference_downscale
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
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized")
        if self.video.is_empty():
            raise ValueError("Connect the clip that drives the output.")

        frames = await context.video_to_frames(self.video, fps=round(self.frame_rate))
        if not frames:
            raise ValueError("The clip has no frames.")
        self._clip_geometry = self._output_geometry(frames[0].size, len(frames))
        output = await self._run_reference(
            context, frames[: self._clip_geometry[2]]
        )
        return await self._video_and_audio(context, output)


class LTXIngredients(_LTXInContext):
    """
    Generates a video with a matching soundtrack that features the people, objects and places from 1 to 8 reference pictures, using the LTX-2.3 ingredients IC-LoRA.
    video, audio, generation, AI, references, ingredients, character, product, ic-lora, ltx, Lightricks

    The pictures are tiled onto one reference sheet, as Phosphene does, and the
    sheet rides along the clip as an IC-LoRA reference. The adapter is a
    community LoRA from DeepBeepMeep/LTX-2.

    Use cases:
    - Put a character and a product into one scene
    - Keep a cast consistent across shots
    """

    model: HFTextToVideo = Field(
        default=HFTextToVideo(repo_id=LTX23_DISTILLED_REPO_ID),
        description="The distilled LTX-2.3 Diffusers checkpoint the ingredients LoRA was trained on.",
    )
    images: list[ImageRef] = Field(
        default=[],
        description=f"1 to {INGREDIENTS_MAX_IMAGES} reference pictures.",
    )
    sheet_description: str = Field(
        default="",
        description="What the reference pictures show, such as 'a red-haired woman, a vintage motorbike'.",
    )
    prompt: str = Field(
        default="",
        description="What happens in the clip, including the sound.",
    )
    lora_strength: float = Field(
        default=1.4,
        description="Strength of the ingredients adapter. 1.4 is the recipe value.",
        ge=0.0,
        le=2.0,
    )

    @classmethod
    def get_title(cls) -> str:
        return "LTX-2.3 Ingredients (IC-LoRA)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["images", "sheet_description", "prompt", "num_frames"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToVideo(
                repo_id=LTX23_DISTILLED_REPO_ID,
                allow_patterns=_DIFFUSERS_REPO_ALLOW_PATTERNS,
            ),
            HFTextToVideo(
                repo_id=LTX23_INGREDIENTS_LORA.repo_id,
                path=LTX23_INGREDIENTS_LORA.path,
            ),
        ]

    def get_model_id(self) -> str:
        return self.model.repo_id or LTX23_DISTILLED_REPO_ID

    def _ic_lora(self) -> ICLoRA:
        return LTX23_INGREDIENTS_LORA

    def _prompt(self) -> str:
        return ingredients_prompt(self.sheet_description, self.prompt)

    async def process(self, context: ProcessingContext) -> LTX25.OutputType:
        images = [image for image in self.images if not image.is_empty()]
        if not 1 <= len(images) <= INGREDIENTS_MAX_IMAGES:
            raise ValueError(
                f"Ingredients take 1 to {INGREDIENTS_MAX_IMAGES} pictures, got {len(images)}."
            )
        self._clip_geometry = LTX25._geometry(self)
        width, height, num_frames = self._clip_geometry
        sheet = compose_ingredients_sheet(
            [await context.image_to_pil(image) for image in images]
        ).resize((width, height), Image.Resampling.LANCZOS)
        output = await self._run_reference(context, [sheet] * num_frames)
        return await self._video_and_audio(context, output)


class LTXUpscale(_LTXInContext):
    """
    Doubles the width and height of a video with the LTX-2.5 pixel upscaler IC-LoRA. Keeps the source length and sound.
    video, upscale, super resolution, x2, video-to-video, ic-lora, ltx, ltx-2.5, Lightricks

    The source rides as the IC-LoRA reference at half the output size. The
    adapter is gated: accept its license on Hugging Face before downloading.

    Use cases:
    - Bring a 768x448 LTX draft to 1536x896
    - Sharpen a low-resolution clip before delivery
    """

    model: HFTextToVideo = Field(
        default=HFTextToVideo(repo_id=LTX25_REPO_ID),
        description="The gated LTX-2.5 Diffusers checkpoint.",
    )
    video: VideoRef = Field(
        default=VideoRef(),
        description="The clip to upscale. Up to 768 pixels on the long side.",
    )
    prompt: str = Field(
        default="high detail, sharp, clean",
        description="Optional description of the clip. It steers detail the source lacks.",
    )
    reference_strength: float = Field(
        default=1.0,
        description="How much of the source to keep. Lower values let the model invent more detail.",
        ge=0.3,
        le=1.0,
    )

    @classmethod
    def get_title(cls) -> str:
        return "LTX-2.5 Upscale x2 (IC-LoRA)"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["video", "reference_strength", "frame_rate"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTextToVideo(
                repo_id=LTX25_REPO_ID,
                allow_patterns=_DIFFUSERS_REPO_ALLOW_PATTERNS,
            ),
            HFTextToVideo(
                repo_id=LTX25_UPSCALE_LORA.repo_id, path=LTX25_UPSCALE_LORA.path
            ),
        ]

    def get_model_id(self) -> str:
        return self.model.repo_id or LTX25_REPO_ID

    def _ic_lora(self) -> ICLoRA:
        return LTX25_UPSCALE_LORA

    @staticmethod
    def upscale_geometry(
        size: tuple[int, int], frame_count: int
    ) -> tuple[int, int, int]:
        """(width, height, num_frames): twice the source, frames padded up to 8n+1."""
        multiple = LTX2_GEOMETRY.dim_multiple
        width, height = (2 * max(multiple, side // multiple * multiple) for side in size)
        if max(width, height) > LTX_UPSCALE_MAX_DIM:
            raise ValueError(
                f"A {size[0]}x{size[1]} clip upscales past {LTX_UPSCALE_MAX_DIM} pixels. "
                f"Upscale takes clips up to {LTX_UPSCALE_MAX_DIM // 2} pixels on the long side."
            )
        if not 1 <= frame_count <= LTX2_GEOMETRY.max_frames:
            raise ValueError(
                f"Upscale takes 1 to {LTX2_GEOMETRY.max_frames} frames, got {frame_count}."
            )
        step = LTX2_GEOMETRY.frame_step
        frames = max(LTX2_GEOMETRY.min_frames, math.ceil((frame_count - 1) / step) * step + 1)
        return width, height, frames

    async def process(self, context: ProcessingContext) -> LTX25.OutputType:
        if self.video.is_empty():
            raise ValueError("Connect the clip to upscale.")
        frames = await context.video_to_frames(self.video, fps=round(self.frame_rate))
        if not frames:
            raise ValueError("The clip has no frames.")
        self._clip_geometry = self.upscale_geometry(frames[0].size, len(frames))
        # LTX needs 8n+1 frames: hold the last frame, then cut it again.
        padded = frames + [frames[-1]] * (self._clip_geometry[2] - len(frames))
        output = await self._run_reference(
            context, padded, conditioning_attention_strength=self.reference_strength
        )
        soundtrack = await source_soundtrack(context, self.video)
        return await self._video_and_audio(
            context, output, keep_frames=len(frames), soundtrack=soundtrack
        )
