from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from pydantic import Field

from nodetool.metadata.types import HFImageSegmentation, ImageRef
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext

if TYPE_CHECKING:
    import PIL.Image
    import torch


# ImageNet statistics used by the RMBG / BiRefNet preprocessing pipelines.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_hex_color(value: str) -> tuple[int, int, int, int] | None:
    """
    Parse a hex color string into an RGBA tuple.

    Accepts ``#RGB``, ``#RGBA``, ``#RRGGBB`` and ``#RRGGBBAA`` with or without
    the leading ``#``. Returns ``None`` for an empty/whitespace-only string so
    callers can treat that as "keep the background transparent".

    Raises:
        ValueError: if the string is non-empty but not a valid hex color.
    """
    text = value.strip().lstrip("#")
    if not text:
        return None

    if len(text) in (3, 4):
        text = "".join(ch * 2 for ch in text)

    if len(text) not in (6, 8):
        raise ValueError(
            f"Invalid background color {value!r}: expected a hex color such as "
            "'#ffffff' (or an empty string for transparency)."
        )

    try:
        channels = [int(text[i : i + 2], 16) for i in range(0, len(text), 2)]
    except ValueError as exc:
        raise ValueError(
            f"Invalid background color {value!r}: not a hexadecimal value."
        ) from exc

    if len(channels) == 3:
        channels.append(255)
    return (channels[0], channels[1], channels[2], channels[3])


def matte_dtype(device: str) -> "torch.dtype":
    """
    Pick a dtype for matting inference.

    RMBG-2.0 / BiRefNet run happily in half precision on CUDA, but half
    precision is fragile on CPU and on MPS for these custom modules, so those
    devices stay in float32.
    """
    import torch

    return torch.float16 if str(device).startswith("cuda") else torch.float32


class BackgroundRemoval(HuggingFacePipelineNode):
    """
    Removes image backgrounds with RMBG / BiRefNet matting models, returning a transparent cutout and its alpha matte.
    image, background-removal, matting, segmentation, alpha, cutout, rmbg, birefnet, huggingface

    These checkpoints ship custom modeling code on the Hub, so they are loaded
    with ``trust_remote_code=True`` — only use repositories you trust.

    Use cases:
    - Cut out products for e-commerce listings on a clean white background
    - Isolate people or pets from photos for compositing and collages
    - Produce transparent PNG assets for design, slides, and thumbnails
    - Generate alpha mattes to drive downstream inpainting or relighting
    - Batch-prepare reference images for image-to-3D or try-on pipelines
    """

    model: HFImageSegmentation = Field(
        default=HFImageSegmentation(repo_id="briaai/RMBG-2.0"),
        title="Model",
        description="The matting model to use. RMBG-2.0 is the strongest general-purpose remover, BiRefNet gives very crisp edges (hair, fur), and RMBG-1.4 is the lightest/fastest option.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Image",
        description="The input image whose background should be removed.",
    )
    background_color: str = Field(
        default="",
        title="Background Color",
        description="Hex color to composite the cutout onto (e.g. '#ffffff'). Leave empty to keep the background transparent.",
    )
    resolution: int = Field(
        default=1024,
        title="Processing Resolution",
        description="Square resolution the image is resized to before inference. 1024 matches how these models were trained; lower is faster, higher can sharpen edges on large images.",
        ge=256,
        le=2048,
    )

    _pipeline: Any = None
    _device: str = "cpu"

    @classmethod
    def get_recommended_models(cls) -> list[HFImageSegmentation]:
        return [
            HFImageSegmentation(
                repo_id="briaai/RMBG-2.0",
                allow_patterns=[
                    "README.md",
                    "*.safetensors",
                    "*.json",
                    "**/*.json",
                    "*.py",
                    "**/*.py",
                ],
            ),
            HFImageSegmentation(
                repo_id="ZhengPeng7/BiRefNet",
                allow_patterns=[
                    "README.md",
                    "*.safetensors",
                    "*.json",
                    "**/*.json",
                    "*.py",
                    "**/*.py",
                ],
            ),
            HFImageSegmentation(
                repo_id="briaai/RMBG-1.4",
                allow_patterns=[
                    "README.md",
                    "*.safetensors",
                    "*.pth",
                    "*.json",
                    "**/*.json",
                    "*.py",
                    "**/*.py",
                ],
            ),
        ]

    @classmethod
    def get_title(cls) -> str:
        return "Background Removal"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "image", "background_color"]

    def required_inputs(self):
        return ["image"]

    def get_model_id(self) -> str:
        return self.model.repo_id

    class OutputType(TypedDict):
        image: ImageRef
        mask: ImageRef

    async def preload_model(self, context: ProcessingContext):
        from transformers import AutoModelForImageSegmentation

        self._device = context.device
        self._pipeline = await self.load_model(
            context=context,
            model_class=AutoModelForImageSegmentation,
            model_id=self.get_model_id(),
            variant=None,
            # RMBG / BiRefNet ship their modeling code in the repo itself.
            trust_remote_code=True,
        )
        if hasattr(self._pipeline, "eval"):
            self._pipeline.eval()

    async def move_to_device(self, device: str):
        if self._pipeline is None:
            return
        self._device = device
        self._pipeline.to(device=device, dtype=matte_dtype(device))

    def preprocess(self, image: "PIL.Image.Image") -> "torch.Tensor":
        """Resize to the model's square input and normalize with ImageNet stats."""
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        tensor = transform(image.convert("RGB")).unsqueeze(0)
        return tensor.to(device=self._device, dtype=matte_dtype(self._device))

    @staticmethod
    def extract_matte(outputs: Any) -> "torch.Tensor":
        """
        Pull the alpha matte out of a matting model's output.

        BiRefNet-style models return a list of supervision maps whose last
        entry is the final prediction; RMBG-1.4 returns a tuple of such lists,
        and a plain tensor is possible too. All shapes collapse to a single
        ``(1, 1, H, W)`` logit tensor here, which is then squashed by sigmoid.
        """
        result = outputs
        while isinstance(result, (list, tuple)):
            if not result:
                raise ValueError("Matting model returned an empty prediction.")
            result = result[-1]
        if not hasattr(result, "sigmoid"):
            raise ValueError(
                f"Unexpected matting model output of type {type(outputs).__name__}."
            )
        return result.sigmoid().float().cpu()

    async def process(self, context: ProcessingContext) -> OutputType:
        import PIL.Image

        if self._pipeline is None:
            raise ValueError("Model not initialized")

        background = parse_hex_color(self.background_color)

        image = await context.image_to_pil(self.image)
        image = image.convert("RGB")

        inputs = self.preprocess(image)
        outputs = await self.run_pipeline_in_thread(inputs)
        matte = self.extract_matte(outputs)

        # (1, 1, H, W) -> (H, W) grayscale, back at the original resolution.
        mask_image = PIL.Image.fromarray(
            (matte.squeeze().clamp(0, 1).numpy() * 255).astype("uint8"), mode="L"
        ).resize(image.size, PIL.Image.Resampling.BILINEAR)

        cutout = image.convert("RGBA")
        cutout.putalpha(mask_image)

        if background is not None:
            plate = PIL.Image.new("RGBA", image.size, background)
            cutout = PIL.Image.alpha_composite(plate, cutout)

        return {
            "image": await context.image_from_pil(cutout),
            "mask": await context.image_from_pil(mask_image),
        }
