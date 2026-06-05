from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from nodetool.metadata.types import HFTimeSeriesForecast, HuggingFaceModel
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext

if TYPE_CHECKING:
    import torch


class TimesFMForecast(HuggingFacePipelineNode):
    """
    Forecasts future values of a univariate time series using Google's TimesFM 2.5.
    time-series, forecasting, prediction, timesfm, zero-shot

    Use cases:
    - Zero-shot forecasting of demand, traffic, or sensor readings
    - Extend a historical series into the future without task-specific training
    - Produce baseline forecasts for anomaly detection
    - Capacity planning from past usage data
    """

    model: HFTimeSeriesForecast = Field(
        default=HFTimeSeriesForecast(
            repo_id="google/timesfm-2.5-200m-transformers",
        ),
        title="Model",
        description="The TimesFM forecasting model. 2.5-200m supports context up to 16,384 points.",
    )
    context_values: list[float] = Field(
        default=[],
        title="Context Values",
        description="Historical series values in chronological order (oldest first, most recent last).",
        json_schema_extra={"expose": "handle"},
    )
    horizon: int = Field(
        default=128,
        ge=1,
        le=1024,
        title="Forecast Horizon",
        description="Number of future steps to forecast. Capped at the model's horizon length (~128 per call).",
    )

    _pipeline: Any = None

    @classmethod
    def get_title(cls) -> str:
        return "TimesFM Forecast"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "context_values", "horizon"]

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HFTimeSeriesForecast(
                repo_id="google/timesfm-2.5-200m-transformers",
                allow_patterns=["*.safetensors", "*.json", "*.txt"],
            ),
        ]

    def required_inputs(self):
        return ["context_values"]

    def get_model_id(self) -> str:
        return self.model.repo_id

    async def preload_model(self, context: ProcessingContext):
        import torch
        from transformers import TimesFm2_5ModelForPrediction

        if not self.model.repo_id:
            raise ValueError("Model ID is required")

        # Forecasting is precision-sensitive and the model is small (~200M),
        # so load in float32 rather than a reduced inference dtype.
        self._pipeline = await self.load_model(
            context=context,
            model_class=TimesFm2_5ModelForPrediction,
            model_id=self.get_model_id(),
            torch_dtype=torch.float32,
        )

    async def process(self, context: ProcessingContext) -> list[float]:
        import torch

        assert self._pipeline is not None, "Model not initialized"
        if not self.context_values:
            raise ValueError("context_values must contain at least one value")

        series = torch.tensor(
            self.context_values,
            dtype=torch.float32,
            device=self._pipeline.device,
        )
        # TimesFM consumes a sequence of 1D tensors (one per series).
        outputs = await self.run_pipeline_in_thread(
            past_values=[series], return_dict=True
        )
        forecast = outputs.mean_predictions[0]
        return forecast[: self.horizon].float().cpu().tolist()
