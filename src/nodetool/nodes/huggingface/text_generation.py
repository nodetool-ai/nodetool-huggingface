from enum import Enum
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
        default=512,
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

    @classmethod
    def get_recommended_models(cls) -> list[HFTextGeneration]:
        return [
            HFTextGeneration(repo_id="nvidia/Nemotron-Orchestrator-8B"),
            HFTextGeneration(repo_id="EssentialAI/rnj-1-instruct"),
            HFTextGeneration(repo_id="EssentialAI/rnj-1"),
            HFTextGeneration(repo_id="meta-llama/Llama-3.1-8B-Instruct"),
            HFTextGeneration(repo_id="arcee-ai/Trinity-Nano-Preview"),
            HFTextGeneration(repo_id="open-thoughts/OpenThinker-Agent-v1"),
            HFTextGeneration(repo_id="Qwen/Qwen3-0.6B"),
            HFTextGeneration(repo_id="Qwen/Qwen3-4B-Instruct-2507"),
            HFTextGeneration(repo_id="Qwen/Qwen2.5-7B-Instruct"),
            HFTextGeneration(repo_id="meta-llama/Llama-3.1-8B"),
            HFTextGeneration(repo_id="allenai/Olmo-3-32B-Think"),
            HFTextGeneration(repo_id="Qwen/Qwen3-1.7B"),
            HFTextGeneration(repo_id="HuggingFaceTB/SmolLM3-3B"),
            HFTextGeneration(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            # Unsloth Qwen3 BitsandBytes 4-bit safetensors (bnb-4bit) models
            HFTextGeneration(repo_id="unsloth/Qwen3-32B-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/Qwen3-30B-A3B-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/Qwen3-14B-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/Qwen3-8B-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/Qwen3-4B-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/Qwen3-1.7B-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/Qwen3-0.6B-bnb-4bit"),
            # (Base) Qwen3 BitsandBytes 4-bit safetensors
            HFTextGeneration(repo_id="unsloth/Qwen3-30B-A3B-Base-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/Qwen3-14B-Base-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/Qwen3-8B-Base-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/Qwen3-4B-Base-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/Qwen3-1.7B-Base-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/Qwen3-0.6B-Base-bnb-4bit"),
            # Unsloth Gemma 3 BitsandBytes 4-bit models
            HFTextGeneration(repo_id="unsloth/gemma-3-270m-it-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/gemma-3-1b-it-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/gemma-3-4b-it-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/gemma-3-12b-it-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/gemma-3-27b-it-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/gemma-3-1b-pt-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/gemma-3-4b-pt-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/gemma-3-12b-pt-bnb-4bit"),
            HFTextGeneration(repo_id="unsloth/gemma-3-27b-pt-bnb-4bit"),
            # Unsloth Ministral 3 BitsandBytes 4-bit models
            HFTextGeneration(repo_id="unsloth/Ministral-3-14B-Instruct-2512-bnb-4bit"),
            HFTextGeneration(
                repo_id="unsloth/Ministral-3-14B-Instruct-2512-unsloth-bnb-4bit"
            ),
            HFTextGeneration(repo_id="unsloth/Ministral-3-14B-Reasoning-2512-bnb-4bit"),
            HFTextGeneration(
                repo_id="unsloth/Ministral-3-14B-Reasoning-2512-unsloth-bnb-4bit"
            ),
            HFTextGeneration(repo_id="unsloth/Ministral-3-8B-Instruct-2512-bnb-4bit"),
            HFTextGeneration(
                repo_id="unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit"
            ),
            HFTextGeneration(repo_id="unsloth/Ministral-3-8B-Reasoning-2512-bnb-4bit"),
            HFTextGeneration(
                repo_id="unsloth/Ministral-3-8B-Reasoning-2512-unsloth-bnb-4bit"
            ),
            HFTextGeneration(repo_id="unsloth/Ministral-3-3B-Instruct-2512-bnb-4bit"),
            HFTextGeneration(
                repo_id="unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit"
            ),
            HFTextGeneration(repo_id="unsloth/Ministral-3-3B-Reasoning-2512-bnb-4bit"),
            HFTextGeneration(
                repo_id="unsloth/Ministral-3-3B-Reasoning-2512-unsloth-bnb-4bit"
            ),
            HFTextGeneration(repo_id="unsloth/Ministral-3-14B-Base-2512-bnb-4bit"),
            HFTextGeneration(
                repo_id="unsloth/Ministral-3-14B-Base-2512-unsloth-bnb-4bit"
            ),
            HFTextGeneration(repo_id="unsloth/Ministral-3-8B-Base-2512-bnb-4bit"),
            HFTextGeneration(
                repo_id="unsloth/Ministral-3-8B-Base-2512-unsloth-bnb-4bit"
            ),
            HFTextGeneration(repo_id="unsloth/Ministral-3-3B-Base-2512-bnb-4bit"),
            HFTextGeneration(
                repo_id="unsloth/Ministral-3-3B-Base-2512-unsloth-bnb-4bit"
            ),
        ]

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
        provider = await get_provider(Provider.HuggingFace)
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
