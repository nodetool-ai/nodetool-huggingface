# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.5] - 2026-08-16

### Fixed

- Six related VRAM leaks that let GPU memory grow until a workflow OOMed:
  - `offload_kind()` now recognizes both `AlignDevicesHook` and `CpuOffload`,
    so model-level offload is no longer reported as "no offload" and a second
    strategy can't be stacked on the first
  - Offload is no longer torn down and rebuilt on every `process()` call —
    all call sites route through `apply_cpu_offload_if_needed()`, a no-op when
    a strategy is already installed
  - Moves to an accelerator go through `move_pipeline_to_device()`, which skips
    the move for offloaded pipelines instead of pulling their weights back onto
    the GPU
  - LoRA/IP-Adapter pipelines are visible to VRAM management again:
    `should_skip_cache()` is replaced by `model_cache_suffix()`, keeping adapter
    variants in distinct but still `ModelManager`-registered cache entries, and
    re-binding an already-bound adapter set is skipped
  - OOM retries release the traceback before reclaiming, so the failed load's
    GPU tensors are actually freed
  - Cache keys include component identity, so an SDXL pipeline assembled with a
    Nunchaku UNet no longer shares a key with the full-precision one
- SepFormer is registered with `ModelManager` instead of loading unregistered
  straight onto the GPU; Whisper's duplicate cache entry is dropped; reclaim now
  happens proactively on a cache miss rather than only after an OOM

## [0.7.4] - 2026-08-15

### Added

- `SpeakerDiarization` node wrapping pyannote.audio
  (`speaker-diarization-3.1`, `community-1`, `segmentation-3.0`). Outputs
  Whisper-compatible `AudioChunk` segments plus a deduped speaker list,
  supports num/min/max speaker hints, and resolves the gated-checkpoint HF
  token from context secrets with an environment fallback
- `SpeechSeparation` and `SpeechEnhancement` nodes on SpeechBrain's SepFormer:
  separation returns one `AudioRef` per speaker (wsj02mix/wsj03mix/whamr16k),
  enhancement denoises via the dns4/wham checkpoints. Sample rate comes from
  the model hparams with a repo-id fallback table; outputs are peak-normalized
- `PoseEstimation` and `VisualizePoseEstimation` nodes: an RT-DETR person
  detector feeds ViTPose for COCO-17 keypoints per person, including MoE
  (`vitpose-plus`) `dataset_index` handling, with PIL skeleton rendering
- `BackgroundRemoval` node for RMBG-2.0 / BiRefNet / RMBG-1.4 matting,
  returning both an RGBA cutout and the grayscale matte, with optional
  compositing onto a hex background color

### Changed

- `pyannote.audio` and `speechbrain` are required dependencies rather than
  optional extras, so diarization and source separation work out of the box
- LTX-2.5 runs its distilled, unguided recipe — 8 distilled sigmas,
  `guidance_scale=1.0`, STG/modality guidance zeroed by feature detection —
  roughly 10x fewer model evaluations than the inherited 40-step CFG-4
  schedule, which also degraded distilled output
- Video geometry is snapped to each model's legal lattice (LTX: 8n+1 frames,
  dimensions divisible by 32; MiniMax H3: 17n+5 frames) with warnings on
  adjustment and clear errors when out of range, instead of failing deep
  inside the pipeline
- MiniMax H3 enforces a joint `width * height * num_frames` budget before any
  model load, turning multi-minute OOMs into immediate, actionable errors

### Fixed

- Pipelines now fit the available VRAM on GPUs below 24GB instead of OOMing:
  weight footprint is measured against free VRAM to pick full device
  placement, model CPU offload, or sequential CPU offload. The text-to-image
  and image-to-image loaders previously moved Flux/SD3-class pipelines to CUDA
  unconditionally and discarded `use_cpu_offload` without installing any
  offload; 13B+ video transformers (Wan T2V/I2V/FLF2V, LTX-2 T2V/I2V) now fall
  back to sequential offload when model-level offload cannot fit them
- `load_model` honors an explicit `device` argument — `device="cpu"` callers
  (Wan, LTX-2, AceStep, LongCat, Flux) leaked the kwarg into `from_pretrained`
  and force-moved the full model to CUDA before offload hooks were installed
- LLMs/VLMs load in half precision; the text-generation and VLM streaming
  paths plus the Qwen2.5-VL and Qwen3-VL nodes previously materialized fp32
  weights (~2x VRAM)
- VAE tiling is enabled by default for Wan video decode, and VAE
  slicing + tiling was added to the LTX-2 and Wan FLF2V nodes
- The CUDA caching allocator is flushed after `run_pipeline_in_thread` so the
  next pipeline in a workflow sees the freed VRAM
- TripoSG loads weights directly in fp16 and skips the full-device move under
  CPU offload, removing a 2x load-time VRAM spike
- The local provider imported `AutoencoderKLWan`/`WanPipeline` and
  `pipeline_progress_callback` from `text_to_video`, where they exist only
  under `TYPE_CHECKING` (or not at all) — the default Wan text-to-video path
  crashed at runtime
- LTX-2.5 repos requested through the provider path route to the `LTX25` node
  so they use the distilled recipe rather than the guided LTX2 defaults
- LoRA adapter names are derived by stripping only the final extension,
  sanitizing to `[A-Za-z0-9_]` and appending a short sha256 suffix, so
  versioned filenames no longer collide onto one adapter name
- Previously bound LoRA weights are unloaded before loading the requested set,
  so repeated runs no longer stack adapters; an empty selection unbinds all
- LoRAs are loaded before `enable_model_cpu_offload()` in the SD/SDXL
  `run_pipeline` — adapters injected under live offload hooks escaped device
  accounting

## [Unreleased] - 2026-07-28

### Added

- `MiniMaxH3` and `MiniMaxH3Reference` nodes for MiniMax-H3, which generates a
  video and its soundtrack together in one denoising loop. `MiniMaxH3` covers
  the `t2va` (text only) and `fl2va` (first and/or last keyframe) workflows,
  `MiniMaxH3Reference` the `ref2va` omni-reference workflow (up to 9 image,
  3 video and 3 audio references). Both output the muxed video and the
  soundtrack as separate outputs
- `video_from_frames_with_audio` for muxing generated frames and a generated
  soundtrack into one MP4
- Current-generation embedding models for feature extraction and sentence
  similarity (Qwen3-Embedding, Granite Embedding R2, gte-modernbert, multilingual-e5)
- ModernBERT and mmBERT for fill-mask
- gte-reranker-modernbert, mxbai-rerank and ms-marco cross-encoders for reranking
- zeroshot-v2.0 and ModernBERT-NLI models for zero-shot text classification;
  multilingual sentiment/emotion models for text classification
- NER and PII models for token classification (bert-base-NER,
  roberta-large-ner-english, bert-small-pii-detection)
- RT-DETRv2 and D-FINE for object detection; Grounding DINO base, MM Grounding
  DINO and LLMDet for zero-shot object detection
- SigLIP 2 for zero-shot image classification; ConvNeXt V2 and updated NSFW
  detectors for image classification
- Panoptic segmentation models (EoMT, Mask2Former, OneFormer) and SegFormer-B5
- Depth-Anything V2 Metric Large variants and Distill-Any-Depth
- V-JEPA 2 checkpoints for video classification
- Distil-Whisper large-v3.5 for speech recognition; speech emotion, keyword,
  language-ID and gender models for audio classification; more CLAP variants
- TrOCR (printed/handwritten) and BLIP-large for image-to-text
- M2M-100 for translation, which covers all 100 languages the node exposes
  rather than T5's English-source-only pairs
- Qwen3.5/Qwen3.6 for text generation, and Qwen3.5/3.6 plus Gemma 4 for
  image-text-to-text
- `scripts/sync_recommended_models.py` to regenerate `recommended_models` in the
  package metadata without a full `nodetool package scan`

### Changed

- `transformers` and `diffusers` are installed from their upstream `main`
  branches instead of released versions, which is where the MiniMax-H3
  integration lives; `av` (PyAV) is now a direct dependency

### Fixed

- Replaced Hugging Face repositories that no longer exist and would fail at
  download time: `llava-hf/llava-v1.5-{7b,13b}` (now `llava-hf/llava-1.5-{7b,13b}-hf`),
  `Qwen/Qwen3-VL-7B-Instruct`, `Qwen/Qwen3-VL-Demo`, `Wan-AI/Wan2.2-T2V-14B-Diffusers`,
  `Wan-AI/Wan2.2-FLF2V-14B-720P-Diffusers`, `andite/pastel-mix`,
  `cloudqi/cqi_visual_question_answering_pt_v0`, `facebook/mms-tts-{jpn,zho}`
  and `sayakpaul/videomae-base-finetuned-ucf101-subset`
- `Wan_FLF2V` defaulted to a Wan 2.2 FLF2V checkpoint that was never published;
  it now defaults to the Wan 2.1 720P checkpoint
- `TextClassifier` recommended models now allow `*.safetensors`, so
  safetensors-only checkpoints download correctly
- `ImageToText` nodes pass an empty prompt so the `image-text-to-text` pipeline
  (which `image-to-text` now resolves to) accepts the image, and decode
  `generated_text` whether it comes back as a string or as chat messages
- `TextClassifier` passes `top_k=None` so its label to score mapping contains
  every label instead of only the winning one
- `Bark` reads the sampling rate reported by the pipeline instead of assuming
  24 kHz

### Removed

- Nodes whose pipeline tasks were dropped in transformers 5 and that could no
  longer load: `Summarization`, `Translation`, `Text2TextGeneration`,
  `QuestionAnswering` (`TableQuestionAnswering` is unaffected),
  `VisualQuestionAnswering` and `Swin2SR`

## [Unreleased] - 2025-12-20

### Added

- Nunchaku model packs for optimized inference
- Sentence transformers support
- FLUX models for image processing
- IP Adapter support for Stable Diffusion
- ControlNet models for SDXL
- Qwen Image models for vision tasks
- Text-to-video functionality
- GGUF model support
- Enhanced HuggingFace pipeline caching mechanism
- HuggingFace search helper and field fixes
- Model reachability checking scripts

### Changed

- Refactored HuggingFace pipeline loading and caching
- Switched to asynchronous HuggingFace caching (HF_FAST_CACHE)
- Implemented async cache resolution with cache key support
- Enhanced Nunchaku integration across multiple nodes
- Optimized pipelines for better performance
- Enhanced HuggingFaceLocalProvider with MPS device support
- Refactored model loading with improved logging
- Updated Flux model defaults
- Optimized dtype handling (automatic BF16 with fallback)
- Refactored CUDA availability checks
- Enhanced dependencies for improved compatibility
- Replaced nonreachable Nunchaku repos with mit-han-lab

### Fixed

- RealESRGAN functionality
- Nunchaku imports on macOS
- Device handling in Nunchaku transformer
- Caching issues in pipeline loading
- Qwen Image model compatibility
- Flux model configuration
- SDXL ControlNet implementation
- Playground support in XL nodes
- Object detection visualization (plt.tight_layout typo)

### Removed

- IP adapter feature (temporarily removed)
- BitsAndBytes dependency for non-Darwin platforms
- MLX dependencies (mlx_lm, mlx_vlm)
- Obsolete "Image To Audio Story" example
- Deprecated Hugging Face models from LORA and SDXL lists
- Deprecated dependencies from pyproject.toml

## [September 2025]

### Added

- Asynchronous pipeline execution across all nodes
- Kokoro TTS model implementation for multilingual TTS
- Wan image-to-video and text-to-video models
- Multiple new model integrations
- Enhanced model metadata and properties
- New return_type methods for image and latent outputs

### Changed

- Standardized logging across all nodes
- Refactored model loading for async operations
- Improved type handling and dependency management
- Updated to use run_pipeline_in_thread for async execution
- Refactored QwenImage model loading
- Improved model path handling
- Enhanced example workflows

### Fixed

- Import problems across modules
- Model loading edge cases
- Pipeline execution issues

### Removed

- Unused GGUF file parameters
- Deprecated model variants

## [May-August 2025]

### Added

- OmniGen model support for image generation
- CogVideoX for text-to-video generation
- Perturbed-Attention Guidance (PAG) pipelines
- Image-to-text models (LoadImageToTextModel)
- AutoPipeline for image-to-image and inpainting
- SAM2 package integration
- Depth estimation capabilities
- Stable Diffusion Latent Upscaler
- VAE Encode/Decode nodes
- Multiple new model variants and configurations
- Enhanced model selection options
- Chroma integration nodes

### Changed

- Switched from Poetry to Hatchling build system
- Refactored model loading parameters for conditional torch_dtype
- Enhanced image generation with PAG parameters
- Improved import organization and structure
- Updated dependency versions:
  - diffusers to 0.34.0
  - transformers to 4.53.2
  - Added peft with minimum 0.15.0
- Renamed classes for clarity (AutoPipelineImg2Img → ImageToImage)
- Enhanced callbacks and model loading

### Fixed

- ControlNet for Apple MPS devices
- Style transfer examples
- Audio examples and processing
- Import paths and dependencies
- Typos in visualization code

### Removed

- Obsolete image assets
- "Generate Music" workflow example
- Stable Diffusion.json example file

## [January 2025]

### Added

- Enhanced model metadata across nodes
- Additional example workflows

### Changed

- Minor refinements to model configurations
- Updated dependencies

### Fixed

- Model compatibility issues

## [October-December 2024]

### Added

- Initial development phase
- Foundation for HuggingFace integration

### Changed

- Early architecture decisions
- Model loading patterns

### Fixed

- Initial bugs and compatibility issues

## [February-April 2025]

### Added

- Package initialization
- PaddleOCR dependencies
- Stable Diffusion scheduler support
- Poetry configuration
- Initial node implementations:
  - Text-to-image
  - Image-to-image
  - Automatic speech recognition
  - Audio classification
  - Translation
- Example workflows for various tasks

### Changed

- Refactored imports from providers to nodes directory
- Updated Poetry lock and dependencies
- Model manager import path updates
- Dependency management improvements

### Removed

- Legacy dependency management files

## [September 2024]

### Added

- Initial HuggingFace integration
- Pipeline-based node architecture
- Core provider implementations
- Basic model support
- Initial example workflows

### Changed

- Refactored for async pipeline execution
- Initial architecture setup
- Provider structure improvements
