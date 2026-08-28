# Nodetool-HuggingFace

HuggingFace nodes for [Nodetool](https://docs.nodetool.ai) — a large integration bringing HuggingFace's Transformers, Diffusers, and related model ecosystems into Nodetool workflows.

## Description

This package ships well over 100 nodes spanning image generation and editing, video and 3D generation, speech and audio, text and document understanding, and multimodal vision-language models. Nodes wrap HuggingFace `transformers` pipelines and `diffusers` pipelines directly, so any compatible model repo on the Hub can generally be used, in addition to the recommended models called out per node.

## Node Categories

### 🎨 Text-to-Image Generation

- **Stable Diffusion** / **Stable Diffusion XL** — classic and XL diffusion pipelines, with LoRA, IP-Adapter and ControlNet support
- **Flux**, **Flux2**, **Flux2 Klein**, **Flux Control** — Black Forest Labs' FLUX family, with Nunchaku (FP16/FP4/INT4) and GGUF quantization and CPU offload for constrained VRAM
- **Chroma** — Flux-based architecture with enhanced attention-based color control
- **Qwen-Image**, **Qwen-Image-Layered** — Alibaba's Qwen-Image, including a mode that decomposes an image into separate RGBA layers
- **Bria**, **Bria FIBO** — commercial-ready generation, including structured-JSON-prompt control with FIBO
- **GLM-Image** — Zhipu AI's GLM-Image
- **Kandinsky 5.0 Image Lite**
- **Text2Image (AutoPipeline)** — automatic pipeline selection for any text-to-image checkpoint
- **LoadTextToImageModel** — loads/validates a model repo for use by downstream nodes

### 🖌️ Image-to-Image, Editing & Upscaling

- **Image to Image / Inpaint / ControlNet (SD & SDXL)** — img2img, inpainting, and ControlNet-guided generation for both Stable Diffusion and SDXL
- **QwenImageEdit**, **FluxFill**, **FluxKontext** — instruction-based image editing/inpainting
- **OmniGen** — multimodal image generation and editing from mixed image/text inputs
- **RealESRGAN**, **Stable Diffusion Upscale**, **Stable Diffusion Latent Upscaler** — super-resolution
- **VAEEncode / VAEDecode** — encode images to/decode latents from a Stable Diffusion VAE
- **BackgroundRemoval** — RMBG-2.0 / BiRefNet / RMBG-1.4 matting, returns a cutout and alpha matte
- **LoRA Selector** / **LoRA Selector XL** — combine up to 5 LoRAs with per-LoRA strength, 60+ pre-configured style LoRAs

### 🎬 Video Generation

- **CogVideoX**, **Wan (T2V/I2V/FLF2V)**, **LTX-Video**, **LTX-2 / LTX-2.5 / LTX-2 Video**, **Kandinsky 5.0 Video** — text-to-video and image-to-video diffusion transformers
- **MiniMax-H3** / **MiniMax-H3 Reference** — joint video + soundtrack generation (5–15s @ 24fps) from a prompt, optional keyframes, or image/video/audio references
- **VideoClassifier** — action/scene recognition (VideoMAE, TimeSformer, V-JEPA 2)

### 🧊 3D Generation

- **Shap-E** (text-to-3D and image-to-3D) — always installable, no extra dependencies
- **Hunyuan3D**, **TripoSR**, **TripoSG**, **Trellis2**, **StableFast3D** — image-to-3D mesh generation, most gated behind optional extras (`hunyuan3d`, `triposg`, `sf3d`, `triposr`, or `all-3d`) since they pull heavy or git-only dependencies

### 🗣️ Speech & Audio

- **Whisper** — speech-to-text with 100+ languages, translation mode, word/sentence timestamps; **ChunksToSRT** converts chunks to subtitle files
- **SpeakerDiarization** — "who spoke when" via pyannote.audio
- **SpeechSeparation** / **SpeechEnhancement** — SpeechBrain SepFormer for cocktail-party separation and denoising
- **AudioFlamingo** — NVIDIA Audio/Music Flamingo for speech/sound/music understanding and Q&A
- **AudioClassifier** / **ZeroShotAudioClassifier** — audio tagging with fixed or custom labels
- **Text-to-Speech**: Bark, KokoroTTS (+ `kokoro-ja` extra for Japanese), TextToSpeech (multi-model), SupertonicTTS (CPU, 44.1kHz), F5TTS (voice cloning, `f5-tts` extra), HiggsAudio
- **Text-to-Audio / Music**: MusicGen, MusicLDM, AudioLDM, AudioLDM2, DanceDiffusion, StableAudio, LongCat-AudioDiT
- **ACE-Step 1.5** — music generation plus audio-to-audio tasks: Cover, Repaint, Extract, Lego, Complete

### 📝 Text Processing

- **Text Generation** — LLM text generation with streaming, quantized (BitsAndBytes 4-bit) and GGUF model support across Qwen3, Llama 3.1, Ministral, Gemma 3 and more
- **Text Classification** / **Zero-Shot Text Classification** — sentiment/topic labeling with fixed or custom labels
- **Token Classification** — NER, POS tagging
- **Fill Mask** — masked-language-model completion
- **Table Question Answering** — natural-language queries over DataFrames (TAPAS, TAPEX)
- **Document Question Answering** — Q&A over document images (PDFs, receipts, forms) combining OCR/LayoutLM/Donut
- **Table Structure Recognition** — recovers HTML table structure from an image (PaddleOCR SLANet, `ocr` extra)
- **Reranker** — cross-encoder relevance ranking of text pairs
- **Sentence Similarity**, **Feature Extraction**, **SplitSentences** — embeddings, similarity scoring, tokenizer-aware sentence splitting
- **TimesFMForecast** — zero-shot univariate time-series forecasting with Google's TimesFM 2.5

### 🖼️ Image & Video Understanding

- **Image Classifier** / **Zero-Shot Image Classifier** (CLIP) — labeling with fixed or custom categories
- **Segmentation**, **SAM2Segmentation**, **MaskGeneration** — semantic and instance segmentation, plus SAM/SAM2/SAM3 automatic mask generation; **FindSegment** / **VisualizeSegmentation** helpers
- **Object Detection** / **Zero-Shot Object Detection** — bounding boxes with fixed or custom labels; **VisualizeObjectDetection** helper
- **Depth Estimation** — monocular depth prediction
- **Pose Estimation** — RT-DETR person detection + ViTPose COCO-17 keypoints, including MoE (`vitpose-plus`); **VisualizePoseEstimation** renders the skeleton
- **Image to Text** — image captioning

### 🎭 Vision-Language / Multimodal

- **ImageTextToText**, **Qwen2.5-VL**, **Qwen3-VL** — visual question answering and image/video reasoning with quantization options
- **LoadImageToTextModel** / **LoadImageTextToTextModel** / **LoadImageToImageModel** / **LoadTextToImageModel** — shared model-loading/validation nodes used by downstream pipeline nodes

## Installation

```bash
pip install nodetool-huggingface
```

Or from source:

```bash
git clone https://github.com/nodetool-ai/nodetool-huggingface.git
cd nodetool-huggingface
pip install -e .
```

### Optional extras

Most nodes work out of the box. A few pull in heavier or gated dependencies and are opt-in:

| Extra | Enables |
|---|---|
| `ocr` | `TableStructureRecognition` (PaddleOCR/PaddlePaddle) |
| `kokoro-ja` | Japanese text for `KokoroTTS` (misaki + pyopenjtalk) |
| `f5-tts` | `F5TTS` voice cloning |
| `hunyuan3d` | `Hunyuan3D` mesh generation |
| `triposg` | Optional flash decoder + background removal for `TripoSG` |
| `sf3d` / `triposr` | Companion deps for `StableFast3D` / `TripoSR` (the packages themselves install from `requirements/*.txt`, see below) |
| `all-3d-pypi` / `all-3d` | Convenience bundles of the above 3D extras |

```bash
pip install "nodetool-huggingface[kokoro-ja]"
pip install "nodetool-huggingface[all-3d-pypi]"
```

Some 3D nodes (`StableFast3D`, `TripoSR`, `Trellis2`) depend on packages that are not published to PyPI. Install those from the pinned requirement files in `requirements/` before using the corresponding node, e.g.:

```bash
pip install -r requirements/sf3d.txt
pip install -r requirements/trellis2.txt
```

## Requirements

- Python 3.11+
- PyTorch 2.9.0 (installed automatically; CUDA build recommended for GPU inference)
- See `pyproject.toml` for the full dependency list
- Some nodes (gated Hub models, pyannote checkpoints) require an `HF_TOKEN`

## Usage Examples

### Text Generation
```python
from nodetool.nodes.huggingface.text_generation import TextGeneration
from nodetool.workflows.processing_context import ProcessingContext

text_gen = TextGeneration(
    model=HFTextGeneration(repo_id="Qwen/Qwen3-8B"),
    prompt="Write a short story about a robot learning to paint",
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
)
result = await text_gen.process(context)
```

### Image Generation with Flux
```python
from nodetool.nodes.huggingface.text_to_image import Flux

flux = Flux(
    prompt="A serene landscape with mountains and a lake at sunset, highly detailed",
    width=1024,
    height=1024,
    num_inference_steps=25,
    seed=42,
)
output = await flux.process(context)
# output.image contains the generated ImageRef
```

### Speech-to-Text Transcription
```python
from nodetool.nodes.huggingface.automatic_speech_recognition import Whisper, WhisperLanguage, Task, Timestamps

whisper = Whisper(
    model=HFAutomaticSpeechRecognition(repo_id="openai/whisper-large-v3"),
    audio=audio_input,
    task=Task.TRANSCRIBE,
    language=WhisperLanguage.ENGLISH,
    timestamps=Timestamps.WORD,
)
result = await whisper.process(context)
```

### Image Classification
```python
from nodetool.nodes.huggingface.image_classification import ImageClassifier

classifier = ImageClassifier(
    model=HFImageClassification(repo_id="google/vit-base-patch16-224"),
    image=image_input,
)
results = await classifier.process(context)
```

## Available Workflow Examples

Pre-built example workflows live in `src/nodetool/examples/nodetool-huggingface/` and can be imported directly into Nodetool:

- **Image to Image** — transform images with Stable Diffusion
- **Upscaling** — image super-resolution
- **Segmentation** — segment images into regions
- **Object Detection** — detect and locate objects in images
- **Depth Estimation** — extract depth information from images
- **Add Subtitles To Video** — transcribe and burn subtitles into a video
- **Audio To Spectrogram** — visualize audio as a spectrogram

## Model Downloads

Models are downloaded from the HuggingFace Hub automatically on first use.

1. Set `HF_TOKEN` (environment variable or Nodetool secrets) for gated models
2. Or authenticate with `huggingface-cli login`
3. Models are cached under `~/.cache/huggingface/` by default

### Gated Models
Some models (FLUX, pyannote checkpoints, etc.) require accepting terms on the Hub:
1. Visit the model page on HuggingFace and accept the license
2. Set `HF_TOKEN` in Nodetool settings

## Performance Tips

### Memory
- Use quantized checkpoints (Nunchaku FP4/INT4, GGUF, BitsAndBytes 4-bit) where a node supports them
- Enable CPU offload for large diffusion/video pipelines
- Prefer smaller model variants when possible

### Speed
- Use a CUDA GPU when available
- Prefer turbo/fast/distilled variants (e.g. `whisper-large-v3-turbo`) when acceptable

## Troubleshooting

**CUDA Out of Memory**
- Enable CPU offload in the node's advanced properties
- Switch to a quantized model variant
- Reduce resolution / inference steps / video length

**Model Not Found / Import Error**
- Check whether the node needs an optional extra installed (see the extras table above)
- Verify the model repo id on the Hub and that `HF_TOKEN` is set for gated repos

**Slow Inference**
- Confirm CUDA is actually being used
- Use a smaller or quantized model, or a turbo/distilled variant

## License

AGPL

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
git clone https://github.com/nodetool-ai/nodetool-huggingface.git
cd nodetool-huggingface
pip install -e .
```

### Adding New Nodes
1. Create a new node class in `src/nodetool/nodes/huggingface/`
2. Inherit from `HuggingFacePipelineNode` (or `BaseNode` for non-pipeline helpers)
3. Implement `preload_model()` and `process()`
4. Add a docstring with use cases and recommended models
5. See `CHANGELOG.md` for recent additions and conventions

## Links & Resources

- [Nodetool Documentation](https://docs.nodetool.ai)
- [HuggingFace Hub](https://huggingface.co/models)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [GitHub Repository](https://github.com/nodetool-ai/nodetool-huggingface)
- [Issue Tracker](https://github.com/nodetool-ai/nodetool-huggingface/issues)
