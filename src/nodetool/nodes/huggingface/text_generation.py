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

    @classmethod
    def get_recommended_models(cls):
        return [
            HFTextGeneration(
                repo_id="gpt2", allow_patterns=["*.json", "*.txt", "*.safetensors"]
            ),
            HFTextGeneration(
                repo_id="distilgpt2",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFTextGeneration(
                repo_id="Qwen/Qwen2-0.5B-Instruct",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            HFTextGeneration(
                repo_id="bigcode/starcoder",
                allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ),
            # GGUF quantized models (detected by .gguf file extension)
            HFTextGeneration(
                repo_id="QuantFactory/Qwen3-4B-v0.4-deepresearch-no-think-4-GGUF",
                path="Qwen3-4B-v0.4-deepresearch-no-think-4.Q4_0.gguf",
            ),
            HFTextGeneration(
                repo_id="QuantFactory/Qwen3-4B-v0.4-deepresearch-no-think-4-GGUF",
                path="Qwen3-4B-v0.4-deepresearch-no-think-4.Q4_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="QuantFactory/Qwen3-4B-v0.4-deepresearch-no-think-4-GGUF",
                path="Qwen3-4B-v0.4-deepresearch-no-think-4.Q5_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="QuantFactory/Qwen3-4B-v0.4-deepresearch-no-think-4-GGUF",
                path="Qwen3-4B-v0.4-deepresearch-no-think-4.Q8_0.gguf",
            ),
            # SmolLM-135M GGUF models - lightweight and fast
            HFTextGeneration(
                repo_id="QuantFactory/SmolLM-135M-GGUF",
                path="SmolLM-135M.Q4_0.gguf",
            ),
            HFTextGeneration(
                repo_id="QuantFactory/SmolLM-135M-GGUF",
                path="SmolLM-135M.Q4_K_M.gguf",
            ),
            # SmolLM2-135M GGUF models - improved version of SmolLM
            HFTextGeneration(
                repo_id="QuantFactory/SmolLM2-135M-GGUF",
                path="SmolLM2-135M.Q4_0.gguf",
            ),
            HFTextGeneration(
                repo_id="QuantFactory/SmolLM2-135M-GGUF",
                path="SmolLM2-135M.Q4_K_M.gguf",
            ),
            # SmolLM2-360M GGUF models - larger improved version
            HFTextGeneration(
                repo_id="QuantFactory/SmolLM2-360M-GGUF",
                path="SmolLM2-360M.Q4_0.gguf",
            ),
            HFTextGeneration(
                repo_id="QuantFactory/SmolLM2-360M-GGUF",
                path="SmolLM2-360M.Q4_K_M.gguf",
            ),
            # Llama 3.1 8B Instruct GGUF models - high-quality instruction-tuned model
            HFTextGeneration(
                repo_id="QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
                path="Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
                path="Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf",
            ),
            # Art-0-8B GGUF models - creative and artistic text generation
            HFTextGeneration(
                repo_id="QuantFactory/Art-0-8B-GGUF",
                path="Art-0-8B.Q4_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="QuantFactory/Art-0-8B-GGUF",
                path="Art-0-8B.Q5_K_M.gguf",
            ),
            # MN-Violet-Lotus-12B GGUF models - large high-performance model
            HFTextGeneration(
                repo_id="QuantFactory/MN-Violet-Lotus-12B-GGUF",
                path="MN-Violet-Lotus-12B.Q4_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="QuantFactory/MN-Violet-Lotus-12B-GGUF",
                path="MN-Violet-Lotus-12B.Q5_K_M.gguf",
            ),
            # NeuralDaredevil-8B-abliterated GGUF models - uncensored model variant
            HFTextGeneration(
                repo_id="QuantFactory/NeuralDaredevil-8B-abliterated-GGUF",
                path="NeuralDaredevil-8B-abliterated.Q4_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="QuantFactory/NeuralDaredevil-8B-abliterated-GGUF",
                path="NeuralDaredevil-8B-abliterated.Q5_K_M.gguf",
            ),
            # Jan-v1-4B GGUF models - Jan's fine-tuned 4B model
            HFTextGeneration(
                repo_id="janhq/Jan-v1-4B-GGUF",
                path="Jan-v1-4B-Q4_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="janhq/Jan-v1-4B-GGUF",
                path="Jan-v1-4B-Q5_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="janhq/Jan-v1-4B-GGUF",
                path="Jan-v1-4B-Q6_K.gguf",
            ),
            HFTextGeneration(
                repo_id="janhq/Jan-v1-4B-GGUF",
                path="Jan-v1-4B-Q8_0.gguf",
            ),
            # OpenAI-20B-NEO uncensored GGUF models - large uncensored GPT model
            HFTextGeneration(
                repo_id="DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf",
                path="OpenAI-20B-NEO-Uncensored2-IQ4_NL.gguf",
            ),
            HFTextGeneration(
                repo_id="DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf",
                path="OpenAI-20B-NEO-Uncensored2-Q5_1.gguf",
            ),
            HFTextGeneration(
                repo_id="DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf",
                path="OpenAI-20B-NEO-CODE-DI-Uncensored-Q8_0.gguf",
            ),
            HFTextGeneration(
                repo_id="DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf",
                path="OpenAI-20B-NEO-HRR-CODE-TRI-Uncensored-IQ4_NL.gguf",
            ),
            HFTextGeneration(
                repo_id="DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf",
                path="OpenAI-20B-NEO-HRR-CODE-TRI-Uncensored-Q5_1.gguf",
            ),
            # Qwen3-Coder-30B-A3B-Instruct GGUF models - large coding model
            HFTextGeneration(
                repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
                path="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
                path="Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
                path="Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf",
            ),
            # ERNIE-4.5-21B-A3B-Thinking GGUF models - large reasoning model with thinking capabilities
            HFTextGeneration(
                repo_id="gabriellarson/ERNIE-4.5-21B-A3B-Thinking-GGUF",
                path="ERNIE-4.5-21B-A3B-Thinking-Q4_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="gabriellarson/ERNIE-4.5-21B-A3B-Thinking-GGUF",
                path="ERNIE-4.5-21B-A3B-Thinking-Q5_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="gabriellarson/ERNIE-4.5-21B-A3B-Thinking-GGUF",
                path="ERNIE-4.5-21B-A3B-Thinking-Q8_0.gguf",
            ),
            # gpt-oss-20b GGUF models - open source GPT 20B model
            HFTextGeneration(
                repo_id="unsloth/gpt-oss-20b-GGUF",
                path="gpt-oss-20b-Q4_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="unsloth/gpt-oss-20b-GGUF",
                path="gpt-oss-20b-Q5_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="unsloth/gpt-oss-20b-GGUF",
                path="gpt-oss-20b-Q8_0.gguf",
            ),
            # WEBGEN-4B-Preview GGUF models - web generation specialized model
            HFTextGeneration(
                repo_id="gabriellarson/WEBGEN-4B-Preview-GGUF",
                path="WEBGEN-4B-Preview-Q4_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="gabriellarson/WEBGEN-4B-Preview-GGUF",
                path="WEBGEN-4B-Preview-Q5_K_M.gguf",
            ),
            HFTextGeneration(
                repo_id="gabriellarson/WEBGEN-4B-Preview-GGUF",
                path="WEBGEN-4B-Preview-Q8_0.gguf",
            ),
            # MiniCPM4.1-8B GGUF model - efficient 8B parameter model
            HFTextGeneration(
                repo_id="openbmb/MiniCPM4.1-8B-GGUF",
                path="MiniCPM4.1-8B-Q4_K_M.gguf",
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
