from nodetool.media.video.video_utils import extract_video_frames
from nodetool.metadata.types import MessageTextContent
from nodetool.metadata.types import MessageImageContent
from nodetool.metadata.types import MessageContent
import asyncio
from typing import Any, AsyncGenerator, TypedDict
from enum import Enum
from pydantic import Field
import logging
from nodetool.config.logging_config import get_logger

logger = get_logger(__name__)

from nodetool.metadata.types import (
    HFImageTextToText,
    HFQwen2_5_VL,
    HFQwen3_VL,
    ImageRef,
    Message,
    Provider,
    VideoRef,
)
from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
from nodetool.providers import get_provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk
from transformers import TextStreamer
from queue import Queue
import threading


class LoadImageTextToTextModel(HuggingFacePipelineNode):
    """
    Loads and validates a Hugging Face image-text-to-text model for use in downstream nodes.
    model-loader, vision-language, multimodal, VLM

    Use cases:
    - Pre-load vision-language models for image understanding tasks
    - Validate model availability before running pipelines
    - Configure model settings for ImageTextToText processing
    """

    repo_id: str = Field(
        default="HuggingFaceTB/SmolVLM-Instruct",
        title="Model ID",
        description="The Hugging Face repository ID for the vision-language model (e.g., SmolVLM, LLaVA variants).",
    )

    async def preload_model(self, context: ProcessingContext):
        if not self.repo_id:
            raise ValueError("Model ID is required")
        self._pipeline = await self.load_pipeline(
            context, "image-text-to-text", self.repo_id
        )

    async def process(self, context: ProcessingContext) -> HFImageTextToText:
        return HFImageTextToText(
            repo_id=self.repo_id,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        )


class ImageTextToText(HuggingFacePipelineNode):
    """
    Generates text responses based on an image and text prompt using vision-language models.
    image, text, visual-question-answering, multimodal, VLM, captioning

    Use cases:
    - Answer questions about image content with detailed explanations
    - Generate comprehensive image descriptions and captions
    - Extract structured information (objects, text, layout) from images
    - Perform OCR-free document understanding via natural language
    - Build multi-turn visual conversations and assistants
    """

    model: HFImageTextToText = Field(
        default=HFImageTextToText(
            repo_id="HuggingFaceTB/SmolVLM-Instruct",
        ),
        title="Model",
        description="The vision-language model to use. SmolVLM is lightweight; LLaVA variants offer different capability levels.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The image to analyze and discuss.",
    )
    prompt: str = Field(
        default="Describe this image.",
        title="Prompt",
        description="Your question or instruction about the image. Be specific for better results.",
    )
    max_new_tokens: int = Field(
        default=256,
        title="Max New Tokens",
        description="Maximum length of the generated response in tokens.",
        ge=1,
    )

    class OutputType(TypedDict):
        text: str | None
        chunk: Chunk

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "quantization", "image", "prompt"]

    @classmethod
    def get_recommended_models(cls):
        return [
            HFImageTextToText(
                repo_id="HuggingFaceTB/SmolVLM-Instruct",
            ),
            HFImageTextToText(
                repo_id="llava-hf/llava-v1.5-13b",
            ),
            HFImageTextToText(
                repo_id="llava-hf/llava-v1.5-7b",
            ),
            HFImageTextToText(
                repo_id="llava-hf/bakLlava-v1-hf",
            ),
            HFImageTextToText(
                repo_id="llava-hf/llava-v1.6-mistral-7b-hf",
            ),
            HFImageTextToText(
                repo_id="llava-hf/llava-v1.6-vicuna-7b-hf",
            ),
            HFImageTextToText(
                repo_id="llava-hf/llava-v1.6-vicuna-13b-hf",
            ),
            HFImageTextToText(
                repo_id="llava-hf/llava-v1.6-34b-hf",
            ),
            HFImageTextToText(
                repo_id="llava-hf/llama3-llava-next-8b-hf",
            ),
            HFImageTextToText(
                repo_id="zai-org/GLM-4.6V-Flash",
            ),
        ]

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    def required_inputs(self):
        return ["image"]

    def _prepare_messages(self) -> list[Message]:
        """Prepare the HF-style messages for the pipeline."""
        return [
            Message(
                role="user",
                content=[
                    MessageImageContent(image=self.image),
                    MessageTextContent(text=self.prompt),
                ],
            )
        ]

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator["ImageTextToText.OutputType", None]:
        provider = await get_provider(Provider.HuggingFace)
        messages = self._prepare_messages()

        full_text = ""
        async for chunk in provider.generate_messages(
            messages=messages,
            model=self.model.repo_id,
            max_tokens=self.max_new_tokens,
            context=context,
            node_id=self.id,
            pipeline_task="image-text-to-text",
        ):
            if chunk.content:
                full_text += chunk.content
            yield {"text": None, "chunk": chunk}

        yield {
            "text": full_text,
            "chunk": Chunk(content="", done=True, content_type="text"),
        }


class Qwen2_5_VLQuantization(str, Enum):
    nf4 = "nf4"
    nf8 = "nf8"
    fp16 = "fp16"


class BaseQwenVL(HuggingFacePipelineNode):
    """
    Base class for Qwen vision-language models supporting image and video understanding.
    image, video, multimodal, VLM, Qwen
    """

    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="An image to analyze. Leave empty if providing video input.",
    )
    video: VideoRef = Field(
        default=VideoRef(),
        title="Input Video",
        description="A video to analyze. Leave empty if providing image input.",
    )
    prompt: str = Field(
        default="Describe this image.",
        title="Prompt",
        description="Your question or instruction about the visual content.",
    )
    min_pixels: int = Field(
        default=0,
        title="Min Pixels",
        description="Minimum pixel count for image resizing. Use 0 for automatic sizing.",
    )
    max_pixels: int = Field(
        default=0,
        title="Max Pixels",
        description="Maximum pixel count for image resizing. Use 0 for automatic sizing.",
    )
    max_new_tokens: int = Field(
        default=128,
        title="Max New Tokens",
        description="Maximum length of the generated response in tokens.",
        ge=1,
    )
    _pipeline: Any = None
    _processor: Any = None

    class OutputType(TypedDict):
        text: str | None
        chunk: Chunk

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "quantization", "image", "prompt"]

    def required_inputs(self):
        return []

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    def _ensure_processor(self):
        if self._processor is None:
            logger.debug("Processor is not loaded")
            raise AssertionError("Processor is not loaded")
        logger.debug("Processor is available")
        return self._processor

    def _ensure_pipeline(self):
        if self._pipeline is None:
            logger.debug("Pipeline is not loaded")
            raise AssertionError("Pipeline is not loaded")
        logger.debug(f"Pipeline is available on device: {self._pipeline.device}")
        return self._pipeline

    async def _prepare_inputs(self, context: ProcessingContext):
        logger.debug("Starting input preparation")
        processor = self._ensure_processor()
        pipeline = self._ensure_pipeline()

        # Prepare content list for the message
        content_list = []
        image_inputs = None
        video_inputs = None

        # Handle images - add placeholder to content and collect PIL images
        if self.image.is_set():
            logger.debug(f"Processing image input: {self.image}")
            pil_image = await context.image_to_pil(self.image)
            image_inputs = [pil_image]
            # Add image placeholder to content
            content_list.append({"type": "image"})
            logger.debug(f"Image converted to PIL, total images: {len(image_inputs)}")
        else:
            logger.debug("No image input provided")

        # Handle videos - add placeholder to content and collect frames
        if self.video.is_set():
            logger.debug(f"Processing video input: {self.video}")
            video_bytes = await context.asset_to_bytes(self.video)
            logger.debug(f"Video bytes loaded, size: {len(video_bytes)} bytes")
            video_frames = await asyncio.to_thread(
                extract_video_frames, video_bytes, fps=1
            )
            video_inputs = [video_frames]
            # Add video placeholder to content
            content_list.append({"type": "video"})
            logger.debug(f"Video frames extracted, total frames: {len(video_frames)}")
        else:
            logger.debug("No video input provided")

        # Add text prompt to content
        logger.debug(
            f"Preparing text prompt: {self.prompt[:100]}..."
            if len(self.prompt) > 100
            else f"Preparing text prompt: {self.prompt}"
        )
        content_list.append({"type": "text", "text": self.prompt})

        # Build message with proper content format for VL models
        messages = [
            {
                "role": "user",
                "content": content_list,
            }
        ]
        logger.debug(f"Message content: {content_list}")

        # Apply chat template and process inputs together
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logger.debug(f"Chat template applied, text length: {len(text)}")

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to device
        logger.debug(f"Moving inputs to device: {pipeline.device}")
        inputs = inputs.to(pipeline.device)

        # Ensure consistent cleanup
        if isinstance(inputs, dict) or hasattr(inputs, "pop"):
            inputs.pop("token_type_ids", None)
            logger.debug("Removed token_type_ids from inputs")

        logger.debug("Input preparation completed")
        return inputs, processor, pipeline

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        logger.debug(
            f"Starting generation process with max_new_tokens={self.max_new_tokens}"
        )
        inputs, processor, pipeline = await self._prepare_inputs(context)

        tokenizer = getattr(processor, "tokenizer", None)
        assert tokenizer is not None, "Tokenizer is not loaded"
        logger.debug("Tokenizer retrieved from processor")

        # Use the same AsyncTextStreamer pattern as in huggingface_local_provider.py
        token_queue: Queue = Queue()

        class AsyncTextStreamer(TextStreamer):
            def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
                super().__init__(tokenizer, skip_prompt, **decode_kwargs)
                self.token_queue = token_queue

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("TextStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]

                if self.skip_prompt and self.next_tokens_are_prompt:
                    self.next_tokens_are_prompt = False
                    return

                text = self.tokenizer.decode(value, skip_special_tokens=True)
                if text:
                    self.token_queue.put(text)

            def end(self):
                self.token_queue.put(None)

        logger.debug("Creating AsyncTextStreamer for token streaming")
        streamer = AsyncTextStreamer(
            tokenizer=tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        full_text = ""
        token_count = 0

        def generate():
            pipeline.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
            )

        logger.debug("Starting generation in background thread")
        thread = threading.Thread(target=generate)
        thread.start()

        logger.debug("Starting to stream tokens")
        try:
            while True:
                await asyncio.sleep(0.01)
                while not token_queue.empty():
                    token = token_queue.get_nowait()
                    if token is None:
                        logger.debug(
                            f"Generation completed, total tokens streamed: {token_count}, final text length: {len(full_text)}"
                        )
                        yield {
                            "text": full_text,
                            "chunk": Chunk(content="", done=True, content_type="text"),
                        }
                        return
                    if not token:
                        continue
                    token_count += 1
                    full_text += token
                    yield {
                        "text": None,
                        "chunk": Chunk(
                            content=token,
                            content_type="text",
                            done=False,
                        ),
                    }
                if not thread.is_alive():
                    while not token_queue.empty():
                        token = token_queue.get_nowait()
                        if token is None:
                            logger.debug(
                                f"Generation completed, total tokens streamed: {token_count}, final text length: {len(full_text)}"
                            )
                            yield {
                                "text": full_text,
                                "chunk": Chunk(
                                    content="", done=True, content_type="text"
                                ),
                            }
                            return
                        if not token:
                            continue
                        token_count += 1
                        full_text += token
                        yield {
                            "text": None,
                            "chunk": Chunk(
                                content=token,
                                content_type="text",
                                done=False,
                            ),
                        }
                    break
        finally:
            thread.join(timeout=1.0)
            logger.debug(
                f"Generation thread joined, total tokens streamed: {token_count}, final text length: {len(full_text)}"
            )


class Qwen2_5_VL(BaseQwenVL):
    """
    Analyzes images and videos using Alibaba's Qwen2.5-VL vision-language model.
    image, video, multimodal, VLM, Qwen, visual-understanding

    Use cases:
    - Understand visual content including objects, text, charts, and layouts
    - Comprehend videos with temporal event tracking and scene changes
    - Localize objects with bounding boxes or point annotations
    - Generate structured output (JSON, tables) from visual data
    - Read and interpret documents, diagrams, and UI screenshots

    **Note:** BNB-4bit variants from Unsloth reduce memory usage significantly.
    """

    model: HFQwen2_5_VL = Field(
        default=HFQwen2_5_VL(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        ),
        title="Model",
        description="The Qwen2.5-VL model variant. Larger models (32B, 72B) offer better accuracy; BNB-4bit variants reduce memory usage.",
    )

    @classmethod
    def get_recommended_models(cls) -> list[HFQwen2_5_VL]:
        # All bnb models from the Unsloth Qwen2.5-VL (All Versions) collection
        bnb_models = [
            "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit",
            "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
            "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
            "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
            "unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit",
            "unsloth/Qwen2.5-VL-32B-Instruct-bnb-4bit",
            "unsloth/Qwen2.5-VL-72B-Instruct-unsloth-bnb-4bit",
            "unsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit",
        ]
        return [
            HFQwen2_5_VL(repo_id="Qwen/Qwen2.5-VL-3B-Instruct"),
            HFQwen2_5_VL(repo_id="Qwen/Qwen2.5-VL-7B-Instruct"),
            HFQwen2_5_VL(repo_id="Qwen/Qwen2.5-VL-32B-Instruct"),
            HFQwen2_5_VL(repo_id="Qwen/Qwen2.5-VL-72B-Instruct"),
            *[HFQwen2_5_VL(repo_id=repo_id) for repo_id in bnb_models],
        ]

    async def preload_model(self, context: ProcessingContext):
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
        )
        import torch

        if not self.model.repo_id:
            logger.error("Model ID is required")
            raise ValueError("Model ID is required")

        logger.debug(f"Loading Qwen2.5-VL model: {self.model.repo_id}")
        logger.debug(
            f"Image resize constraints: min_pixels={self.min_pixels}, max_pixels={self.max_pixels}"
        )

        logger.debug("Loading Qwen2_5_VLForConditionalGeneration model")
        self._pipeline = await self.load_model(
            context,
            Qwen2_5_VLForConditionalGeneration,
            self.model.repo_id,
            device_map="auto",
        )
        logger.debug(f"Model loaded successfully on device: {self._pipeline.device}")

        logger.debug("Loading AutoProcessor")
        self._processor = await self.load_model(
            context,
            AutoProcessor,
            self.model.repo_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        logger.debug("Processor loaded successfully")


class Qwen3_VLQuantization(str, Enum):
    nf4 = "nf4"
    nf8 = "nf8"
    fp16 = "fp16"


class Qwen3_VL(BaseQwenVL):
    """
    Analyzes images and videos using Alibaba's next-generation Qwen3-VL vision-language model.
    image, video, multimodal, VLM, Qwen, visual-reasoning, thinking

    Use cases:
    - Advanced visual reasoning across images and video content
    - Instruction-following with improved spatial-temporal grounding
    - Complex multi-step visual analysis with chain-of-thought reasoning
    - Document, chart, and diagram understanding with enhanced accuracy
    - Long-context visual conversations with memory

    **Note:** Thinking variants include extended reasoning capabilities. BNB-4bit variants reduce memory usage.
    """

    model: HFQwen3_VL = Field(
        default=HFQwen3_VL(
            repo_id="Qwen/Qwen3-VL-4B-Instruct",
        ),
        title="Model",
        description="The Qwen3-VL model variant. Thinking variants add chain-of-thought; BNB-4bit reduces memory; larger models improve accuracy.",
    )

    @classmethod
    def get_recommended_models(cls) -> list[HFQwen3_VL]:
        """Return recommended Qwen3-VL models including official and Unsloth Bitsandbytes variants."""
        return [
            # Official Qwen3-VL instruct and thinking models including new variants
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-2B-Instruct"),
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-2B-Thinking"),
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-4B-Instruct"),
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-4B-Thinking"),
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-7B-Instruct"),
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-8B-Instruct"),
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-8B-Thinking"),
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-30B-A3B-Instruct"),
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-30B-A3B-Thinking"),
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-32B-Instruct"),
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-32B-Thinking"),
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-235B-A22B-Instruct"),
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-235B-A22B-Thinking"),
            # Demo space
            HFQwen3_VL(repo_id="Qwen/Qwen3-VL-Demo"),
            # Unsloth Bitsandbytes (bnb) Qwen3-VL instruct/thinking models - common 2B/4B/8B/32B (bnb-4bit, unsloth-bnb-4bit, etc.)
            HFQwen3_VL(repo_id="unsloth/Qwen3-VL-2B-Instruct-bnb-4bit"),
            HFQwen3_VL(repo_id="unsloth/Qwen3-VL-2B-Thinking-bnb-4bit"),
            HFQwen3_VL(repo_id="unsloth/Qwen3-VL-4B-Instruct-bnb-4bit"),
            HFQwen3_VL(repo_id="unsloth/Qwen3-VL-4B-Thinking-bnb-4bit"),
            HFQwen3_VL(repo_id="unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"),
            HFQwen3_VL(repo_id="unsloth/Qwen3-VL-8B-Thinking-bnb-4bit"),
            HFQwen3_VL(repo_id="unsloth/Qwen3-VL-32B-Instruct-bnb-4bit"),
            HFQwen3_VL(repo_id="unsloth/Qwen3-VL-32B-Thinking-bnb-4bit"),
        ]

    async def preload_model(self, context: ProcessingContext):
        from transformers import (
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
        )
        import torch

        if not self.model.repo_id:
            logger.error("Model ID is required")
            raise ValueError("Model ID is required")

        logger.debug(f"Loading Qwen3-VL model: {self.model.repo_id}")
        logger.debug(
            f"Image resize constraints: min_pixels={self.min_pixels}, max_pixels={self.max_pixels}"
        )

        logger.debug("Loading Qwen3VLForConditionalGeneration model")
        self._pipeline = await self.load_model(
            context,
            Qwen3VLForConditionalGeneration,
            self.model.repo_id,
            device_map="auto",
            # attn_implementation="sdpa",
        )
        logger.debug(f"Model loaded successfully on device: {self._pipeline.device}")

        logger.debug("Loading AutoProcessor")
        self._processor = await self.load_model(
            context,
            AutoProcessor,
            self.model.repo_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        logger.debug("Processor loaded successfully")
