from nodetool.metadata.types import HFDepthEstimation, ImageRef
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext


from pydantic import Field


class DepthEstimation(HuggingFacePipelineNode):
    """
    Generates depth maps from single RGB images using monocular depth estimation models.
    image, depth-estimation, 3D, huggingface, computer-vision

    Use cases:
    - Create depth maps for 3D modeling and scene reconstruction
    - Enable augmented reality applications with depth awareness
    - Improve robotic navigation and obstacle detection
    - Enhance scene understanding in autonomous vehicles
    - Generate depth-based visual effects for images and video
    """

    model: HFDepthEstimation = Field(
        default=HFDepthEstimation(repo_id="LiheYoung/depth-anything-base-hf"),
        title="Model",
        description="The depth estimation model to use. Depth-Anything V2 models offer state-of-the-art accuracy; DPT-large is a reliable alternative.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Image",
        description="The input RGB image to estimate depth from. Any standard image format is supported.",
    )

    @classmethod
    def get_recommended_models(cls) -> list[HFDepthEstimation]:
        return [
            HFDepthEstimation(
                repo_id="depth-anything/Depth-Anything-V2-Small-hf",
            ),
            HFDepthEstimation(
                repo_id="depth-anything/Depth-Anything-V2-Base-hf",
            ),
            HFDepthEstimation(
                repo_id="depth-anything/Depth-Anything-V2-Large-hf",
            ),
            HFDepthEstimation(
                repo_id="Intel/dpt-large",
                allow_patterns=[
                    "README.md",
                    "*.safetensors",
                    "*.json",
                    "**/*.json",
                    "txt",
                ],
            ),
        ]

    def required_inputs(self):
        return ["image"]

    @classmethod
    def get_title(cls) -> str:
        return "Depth Estimation"

    def get_model_id(self):
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context, "depth-estimation", self.get_model_id(), device=context.device
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.model.to(device)  # type: ignore

    async def process(self, context: ProcessingContext) -> ImageRef:
        assert self._pipeline is not None
        image = await context.image_to_pil(self.image)
        result = await self.run_pipeline_in_thread(image)
        depth_map = result["depth"]  # type: ignore
        return await context.image_from_pil(depth_map)  # type: ignore
