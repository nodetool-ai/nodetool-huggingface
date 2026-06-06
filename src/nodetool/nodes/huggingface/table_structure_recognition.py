from __future__ import annotations

from typing import Any

from pydantic import Field

from nodetool.metadata.types import (
    HFTableStructureRecognition,
    HuggingFaceModel,
    ImageRef,
)
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext


class TableStructureRecognition(HuggingFacePipelineNode):
    """
    Recognizes the HTML structure of a table in an image using PaddlePaddle's SLANet.
    table, structure, recognition, ocr, document, slanet

    Use cases:
    - Reconstruct table layout (rows, columns, spans) from scanned documents
    - Convert table images into HTML skeletons for downstream parsing
    - Preprocess tables for cell-level OCR pipelines
    - Digitize tabular data from screenshots and photographs
    """

    model: HFTableStructureRecognition = Field(
        default=HFTableStructureRecognition(
            repo_id="PaddlePaddle/SLANet_plus_safetensors",
        ),
        title="Model",
        description="The table structure recognition model (SLANet / SLANet_plus).",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Image",
        description="Input image containing a single table.",
    )

    _pipeline: Any = None
    _processor: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "Table Structure Recognition"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "image"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTableStructureRecognition(
                repo_id="PaddlePaddle/SLANet_plus_safetensors",
                allow_patterns=["*.safetensors", "*.json"],
            ),
        ]

    def required_inputs(self):
        return ["image"]

    def get_model_id(self) -> str:
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        import torch
        from transformers import AutoImageProcessor, AutoModelForTableRecognition

        if not self.model.repo_id:
            raise ValueError("Model ID is required")

        # SLANet is a lightweight CPU-friendly model; float32 keeps the
        # structure decoder numerically stable.
        self._pipeline = await self.load_model(
            context=context,
            model_class=AutoModelForTableRecognition,
            model_id=self.get_model_id(),
            torch_dtype=torch.float32,
        )
        self._processor = await self.load_model(
            context=context,
            model_class=AutoImageProcessor,
            model_id=self.get_model_id(),
        )

    async def process(self, context: ProcessingContext) -> str:
        assert self._pipeline is not None, "Model not initialized"
        assert self._processor is not None, "Processor not initialized"

        image = await context.image_to_pil(self.image)
        inputs = self._processor(images=image, return_tensors="pt").to(
            self._pipeline.device
        )
        outputs = await self.run_pipeline_in_thread(**inputs)

        results = self._processor.post_process_table_recognition(outputs)
        result = results[0] if isinstance(results, (list, tuple)) else results
        if isinstance(result, dict):
            structure = result.get("structure")
        else:
            structure = getattr(result, "structure", None)
        if isinstance(structure, (list, tuple)):
            structure = "".join(str(token) for token in structure)
        return structure or ""
