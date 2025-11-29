from nodetool.metadata.types import HFTextGeneration, Message, Provider
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.providers import get_provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk
from typing import AsyncGenerator, TypedDict

from pydantic import Field


class TextGeneration(HuggingFacePipelineNode):
    """
    Generates text based on a given prompt.
    text, generation, natural language processing

    Use cases:
    - Creative writing assistance
    - Automated content generation
    - Chatbots and conversational AI
    - Code generation and completion
    """

    model: HFTextGeneration = Field(
        default=HFTextGeneration(),
        title="Model ID on Huggingface",
        description="The model ID to use for the text generation. Supports both regular models and GGUF quantized models (detected by .gguf file extension).",
    )
    prompt: str = Field(
        default="",
        title="Prompt",
        description="The input text prompt for generation",
    )
    max_new_tokens: int = Field(
        default=50,
        title="Max New Tokens",
        description="The maximum number of new tokens to generate",
    )
    temperature: float = Field(
        default=1.0,
        title="Temperature",
        description="Controls randomness in generation. Lower values make it more deterministic.",
        ge=0.0,
        le=2.0,
    )
    top_p: float = Field(
        default=1.0,
        title="Top P",
        description="Controls diversity of generated text. Lower values make it more focused.",
        ge=0.0,
        le=1.0,
    )
    do_sample: bool = Field(
        default=True,
        title="Do Sample",
        description="Whether to use sampling or greedy decoding",
    )

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    class OutputType(TypedDict):
        text: str | None
        chunk: Chunk

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "prompt"]

    def _provider_model_id(self) -> str:
        """Return provider-formatted identifier, handling optional GGUF paths."""
        if not self.model.repo_id:
            raise ValueError("Please select a model")
        if self.model.path:
            return f"{self.model.repo_id}:{self.model.path}"
        return self.model.repo_id

    def _build_provider_messages(self) -> list[Message]:
        prompt = self.prompt.strip()
        if not prompt:
            raise ValueError("Prompt must not be empty.")
        return [Message(role="user", content=prompt)]

    async def preload_model(self, context: ProcessingContext):
        # Models load lazily through the HuggingFace provider.
        _ = context

    async def _gen_process_via_provider(
        self, context: ProcessingContext
    ) -> AsyncGenerator["TextGeneration.OutputType", None]:
        provider = get_provider(Provider.HuggingFace)
        messages = self._build_provider_messages()
        model_id = self._provider_model_id()
        full_text = ""

        async for item in provider.generate_messages(
            messages=messages,
            model=model_id,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            context=context,
            node_id=self.id,
        ):
            if not isinstance(item, Chunk):
                continue
            if item.content:
                full_text += item.content
            yield {"text": None, "chunk": item}

        yield {
            "text": full_text,
            "chunk": Chunk(content="", done=True, content_type="text"),
        }

    async def move_to_device(self, device: str):
        _ = device

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator["OutputType", None]:
        """Stream text generation with both chunk and final text outputs."""
        async for payload in self._gen_process_via_provider(context):
            yield payload

    async def process(self, context: ProcessingContext) -> str:
        """Non-streaming version for backwards compatibility."""
        full_text = ""
        async for payload in self.gen_process(context):
            text = payload.get("text")
            if text is not None:
                full_text = text
        return full_text
