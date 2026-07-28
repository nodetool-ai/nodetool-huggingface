# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-07-28

### Added

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
