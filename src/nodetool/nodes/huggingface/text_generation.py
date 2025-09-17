from nodetool.metadata.types import HFTextGeneration
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk
from nodetool.types.job import JobUpdate
from typing import AsyncGenerator, Any
import torch

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
    
    @classmethod
    def return_type(cls):
        return {
            "text": str,
            "chunk": Chunk,
        }

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

    def _is_gguf_model(self) -> bool:
        """Check if the model is a GGUF model based on filename."""
        return self.model.path and self.model.path.lower().endswith('.gguf')

    async def _load_gguf_model(self, context: ProcessingContext):
        """Load text generation model with GGUF quantization."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        if not self.model.path:
            raise ValueError("GGUF model path is required")
        
        # Use base class load_model method for caching with GGUF support
        model = await self.load_model(
            context,
            AutoModelForCausalLM,
            self.model.repo_id,
            torch_dtype=torch.float32,  # GGUF models are dequantized to fp32
            path=self.model.path,
            device_map="auto",
            gguf_file=self.model.path
        )
        
        # Load tokenizer separately (also use caching)
        tokenizer = await self.load_model(
            context,
            AutoTokenizer,
            self.model.repo_id,
            path=self.model.path,
            gguf_file=self.model.path
        )
        
        # Create pipeline manually - don't specify device when using device_map="auto"
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

    async def preload_model(self, context: ProcessingContext):
        if self._is_gguf_model():
            await self._load_gguf_model(context)
        else:
            self._pipeline = await self.load_pipeline(
                context, "text-generation", self.model.repo_id
            )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            # For GGUF models, move both model and tokenizer components
            if hasattr(self._pipeline, 'model') and hasattr(self._pipeline.model, 'to'):
                self._pipeline.model.to(device)  # type: ignore
            if hasattr(self._pipeline, 'tokenizer') and hasattr(self._pipeline.tokenizer, 'to'):
                try:
                    self._pipeline.tokenizer.to(device)  # type: ignore
                except AttributeError:
                    pass  # Some tokenizers don't support .to() method

    async def gen_process(self, context: ProcessingContext) -> AsyncGenerator[tuple[str, Any], None]:
        """Stream text generation with both chunk and final text outputs."""
        assert self._pipeline is not None
        
        # Use TextStreamer for streaming output
        from transformers import TextStreamer
        import asyncio
        import threading
        from queue import Queue
        
        # Create a queue to collect streamed tokens
        token_queue = Queue()
        full_text = ""
        
        class AsyncTextStreamer(TextStreamer):
            def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
                super().__init__(tokenizer, skip_prompt, **decode_kwargs)
                self.token_queue = token_queue
                
            def put(self, value):
                """Override put to send tokens to queue instead of stdout"""
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("TextStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]
                
                if self.skip_prompt and self.next_tokens_are_prompt:
                    self.next_tokens_are_prompt = False
                    return
                
                # Decode the token
                text = self.tokenizer.decode(value, skip_special_tokens=True)
                if text:
                    self.token_queue.put(text)
                    
            def end(self):
                """Signal end of generation"""
                self.token_queue.put(None)  # Sentinel value
        
        # Create the streaming tokenizer
        streamer = AsyncTextStreamer(
            self._pipeline.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Run generation in a separate thread
        def generate():
            try:
                self._pipeline(
                    self.prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    streamer=streamer,
                    return_full_text=False
                )
            except Exception as e:
                token_queue.put(f"Error: {e}")
                token_queue.put(None)
        
        # Start generation in background thread
        thread = threading.Thread(target=generate)
        thread.start()
        
        # Stream tokens as they become available
        try:
            while True:
                # Check queue with timeout to avoid blocking
                try:
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                    
                    # Non-blocking queue check
                    while not token_queue.empty():
                        token = token_queue.get_nowait()
                        if token is None:  # Sentinel value indicating end
                            # Yield final complete text
                            yield "text", full_text
                            return
                        
                        # Accumulate full text
                        full_text += token
                        
                        # Yield chunk for streaming
                        yield "chunk", Chunk(
                            content=token,
                            done=False
                        )
                        
                except Exception:
                    continue
                    
                # Check if thread is still alive
                if not thread.is_alive():
                    # Drain any remaining tokens
                    while not token_queue.empty():
                        token = token_queue.get_nowait()
                        if token is not None:
                            full_text += token
                            yield "chunk", Chunk(
                                content=token,
                                done=False
                            )
                    
                    # Send final chunk and complete text
                    yield "chunk", Chunk(
                        content="",
                        done=True
                    )
                    yield "text", full_text
                    break
                    
        finally:
            # Ensure thread completes
            thread.join(timeout=1.0)

    async def process(self, context: ProcessingContext) -> str:
        """Non-streaming version for backwards compatibility."""
        full_text = ""
        async for slot_name, value in self.gen_process(context):
            if slot_name == "text":
                full_text = value
                break
        return full_text
