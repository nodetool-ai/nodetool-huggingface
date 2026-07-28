from pydantic import Field
from nodetool.metadata.types import (
    ImageRef,
    HFImageToText,
)
from nodetool.nodes.huggingface.huggingface_pipeline import (
    HuggingFacePipelineNode,
    extract_generated_text,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeUpdate


class ImageToText(HuggingFacePipelineNode):
    """
    Generates descriptive text captions from images using vision-language models.
    image, text, captioning, vision-language, accessibility

    Use cases:
    - Generate natural language descriptions of image content
    - Create alt-text for web accessibility compliance
    - Build automatic image cataloging and search systems
    - Enable content discovery through text-based image queries
    """

    model: HFImageToText = Field(
        default=HFImageToText(),
        title="Model",
        description="The image captioning model. BLIP variants offer good quality/speed balance.",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        title="Input Image",
        description="The image to generate a caption for.",
    )
    max_new_tokens: int = Field(
        default=1024,
        title="Max New Tokens",
        description="Maximum length of the generated caption in tokens.",
    )

    @classmethod
    def get_recommended_models(cls):
        return [
            HFImageToText(
                repo_id="Salesforce/blip-image-captioning-base",
                allow_patterns=["*.bin", "*.json", "*.txt"],
            ),
            HFImageToText(
                repo_id="Salesforce/blip-image-captioning-large",
                allow_patterns=["*.bin", "*.json", "*.txt"],
            ),
            HFImageToText(
                repo_id="nlpconnect/vit-gpt2-image-captioning",
                allow_patterns=["*.bin", "*.json", "*.txt"],
            ),
            HFImageToText(
                repo_id="microsoft/git-base-coco",
                allow_patterns=["*.bin", "*.json", "*.txt"],
            ),
        ]

    def required_inputs(self):
        return ["image"]

    @classmethod
    def get_title(cls) -> str:
        return "HF Image Captioning"

    async def preload_model(self, context: ProcessingContext):
        self._pipeline = await self.load_pipeline(
            context=context,
            pipeline_task="image-to-text",
            model_id=self.model.repo_id,
        )

    async def move_to_device(self, device: str):
        if self._pipeline is not None:
            self._pipeline.model.to(device)

    async def process(self, context: ProcessingContext) -> str:
        assert self._pipeline is not None
        image = await context.image_to_pil(self.image)
        # "image-to-text" is resolved to the image-text-to-text pipeline, which
        # raises when handed an image without accompanying text. An empty prompt
        # keeps the captioning behaviour while satisfying that requirement.
        result = await self.run_pipeline_in_thread(
            images=image, text="", max_new_tokens=self.max_new_tokens
        )
        return extract_generated_text(result)


# class MiniCPM(HuggingFacePipelineNode):
#     """
#     Answers questions about images.
#     image, text, question answering, multimodal

#     Use cases:
#     - Image content analysis
#     - Automated image captioning
#     - Visual information retrieval
#     - Accessibility tools for visually impaired users
#     """

#     model: HFMiniCPM = Field(
#         default=HFMiniCPM(),
#         title="Model ID on Huggingface",
#         description="The model ID to use for visual question answering",
#     )
#     image: ImageRef = Field(
#         default=ImageRef(),
#         title="Image 1",
#         description="The image to analyze",
#     )
#     system_prompt: str = Field(
#         default="",
#         title="System Prompt",
#         description="The system prompt to use for the model",
#     )
#     question: str = Field(
#         default="",
#         title="Question",
#         description="The question to be answered about the image",
#     )
#     sampling: bool = Field(
#         default=True,
#         title="Sampling",
#         description="Whether to use sampling or beam search",
#     )
#     temperature: float = Field(
#         default=0.7,
#         title="Temperature",
#         description="The temperature to use for sampling",
#     )

#     _model: AutoModel | None = None
#     _tokenizer: AutoTokenizer | None = None

#     @classmethod
#     def get_recommended_models(cls):
#         return [
#             HFMiniCPM(
#                 repo_id="openbmb/MiniCPM-V-2_6",
#             ),
#             HFMiniCPM(
#                 repo_id="openbmb/MiniCPM-V-2_6-int4",
#             ),
#         ]

#     async def preload_model(self, context: ProcessingContext):
#         self._model = await self.load_model(
#             context=context,
#             model_id=self.model.repo_id,
#             model_class=AutoModel,
#             trust_remote_code=True,
#             attn_implementation="sdpa",
#             torch_dtype=torch.bfloat16,
#             variant=None,
#         )
#         self._model.eval()
#         self._tokenizer = await self.load_model(
#             context=context,
#             model_id=self.model.repo_id,
#             model_class=AutoTokenizer,
#             trust_remote_code=True,
#         )

#     async def move_to_device(self, device: str):
#         # self._model.to(device)
#         pass

#     async def process(self, context: ProcessingContext) -> str:
#         assert self._model is not None
#         assert self._tokenizer is not None
#         image = await context.image_to_pil(self.image)

#         msgs = [
#             {
#                 "role": "user",
#                 "content": self.question,
#             }
#         ]

#         res = self._model.chat(
#             image=image,
#             msgs=msgs,
#             tokenizer=self._tokenizer,
#             sampling=self.sampling,
#             temperature=self.temperature,
#             system_prompt=self.system_prompt,
#         )
#         generated_text = ""
#         for tok in res:
#             generated_text += tok
#         return generated_text
