from __future__ import annotations

from typing import Any

from nodetool.metadata.types import HuggingFaceModel
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext

from pydantic import Field


class Text2TextGeneration(HuggingFacePipelineNode):
    """
    Transforms input text into output text using seq2seq models (T5, FLAN-T5, mT5).
    text, generation, NLP, seq2seq, T5, FLAN-T5, translation, QA

    Use cases:
    - Instruction-following tasks with FLAN-T5 (e.g. "Translate to French: Hello")
    - Grammar correction and text reformatting
    - Question answering in generative form
    - Code generation and transformation
    - Cross-lingual text generation with multilingual models (mT5)
    """

    model: HuggingFaceModel = Field(
        default=HuggingFaceModel(repo_id="google/flan-t5-base"),
        title="Model",
        description="The seq2seq model to use. FLAN-T5 variants are instruction-tuned and handle diverse tasks; T5 is versatile; mT5 supports 100+ languages.",
    )
    text: str = Field(
        default="",
        title="Input Text",
        description="The input text or instruction. For FLAN-T5 prefix with task description (e.g. 'Summarize: ...', 'Translate to German: ...').",
    )
    max_new_tokens: int = Field(
        default=200,
        title="Max New Tokens",
        description="Maximum number of tokens to generate in the output.",
        ge=1,
        le=2048,
    )
    temperature: float = Field(
        default=1.0,
        title="Temperature",
        description="Controls randomness. Lower values (0.1-0.5) give more focused outputs; higher values (>1.0) increase creativity.",
        ge=0.0,
        le=2.0,
    )
    do_sample: bool = Field(
        default=False,
        title="Do Sample",
        description="Enable sampling. Disable for deterministic beam-search decoding.",
    )

    _pipeline: Any = None

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return [
            HuggingFaceModel(
                repo_id="google/flan-t5-small",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="google/flan-t5-base",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="google/flan-t5-large",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="google/flan-t5-xl",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="google/flan-t5-xxl",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="google/t5-v1_1-base",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="google/t5-v1_1-large",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="google/mt5-base",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json", "*.txt"],
            ),
            HuggingFaceModel(
                repo_id="google/mt5-large",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json", "*.txt"],
            ),
            HuggingFaceModel(
                repo_id="Salesforce/codet5-base",
                allow_patterns=["README.md", "*.bin", "*.json", "**/*.json"],
            ),
            HuggingFaceModel(
                repo_id="Salesforce/codet5p-220m",
                allow_patterns=["README.md", "*.safetensors", "*.json", "**/*.json"],
            ),
        ]

    def required_inputs(self):
        return ["text"]

    @classmethod
    def get_title(cls) -> str:
        return "Text-to-Text Generation"

    def get_model_id(self):
        return self.model.repo_id

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.model.to(device)

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context=context,
            pipeline_task="text2text-generation",
            model_id=self.get_model_id(),
            device=context.device,
        )

    async def process(self, context: ProcessingContext) -> str:
        assert self._pipeline is not None
        result = await self.run_pipeline_in_thread(
            self.text,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if self.do_sample else None,
            do_sample=self.do_sample,
        )
        return str(result[0]["generated_text"])
