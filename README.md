# Nodetool-HuggingFace

HuggingFace nodes for Nodetool - A comprehensive integration that brings state-of-the-art AI models to your workflows.

## Description

This package provides a rich set of HuggingFace nodes for integration with Nodetool, allowing you to build powerful AI workflows using cutting-edge models. With support for over 25 different model types, you can create sophisticated pipelines for text, image, audio, and multimodal processing.

## Node Categories

### 🎨 Image Generation

#### Text-to-Image Nodes
- **Stable Diffusion** - Generate high-quality images from text prompts using Stable Diffusion models
  - Custom width/height settings (256-1024px)
  - Configurable inference steps and guidance scale
  - Support for negative prompts
  - Use cases: Art creation, concept visualization, content generation

- **Stable Diffusion XL** - Enhanced image generation with SDXL models
  - Higher resolution outputs (up to 1024px)
  - Improved image quality and detail
  - Support for IP adapters and LoRA models
  - Use cases: Marketing materials, game assets, interior design concepts

- **Flux** - Next-generation image generation with memory-efficient quantization
  - Supports *schnell* (fast) and *dev* (high-quality) variants
  - Nunchaku quantization (FP16, FP4, INT4) for reduced VRAM usage
  - CPU offload support for large models
  - Configurable max_sequence_length for prompt complexity
  - Use cases: High-fidelity image generation with limited hardware

- **Flux Control** - Controlled image generation with depth/canny guidance
  - Depth-aware and edge-guided generation
  - Control image input for structural guidance
  - Quantization support (FP16, FP4, INT4)
  - Use cases: Controlled composition, maintaining structure while changing style

- **Chroma** - Flux-based model with advanced attention masking
  - Professional-quality color control
  - Attention slicing for memory optimization
  - Use cases: Professional photography effects, precise color grading

- **Qwen-Image** - High-quality general-purpose text-to-image generation
  - Nunchaku quantization support
  - True CFG scale control
  - Use cases: General-purpose image generation, quick prototyping

- **Text2Image (AutoPipeline)** - Automatic pipeline selection for any text-to-image model
  - Auto-detects best pipeline for given model
  - Flexible generation without pipeline-specific knowledge
  - Use cases: Testing different models, rapid prototyping

#### Image-to-Image Transformation
- **Image to Image** - Transform existing images using Stable Diffusion
  - Strength parameter controls transformation amount
  - Support for style transfer and image variations
  - Use cases: Style transfer, image enhancement, creative remixing

### 🗣️ Speech & Audio Processing

#### Audio Classification
- **Audio Classifier** - Classify audio into predefined categories
  - Recommended models:
    - `MIT/ast-finetuned-audioset-10-10-0.4593`
    - `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
  - Use cases: Music genre classification, speech detection, environmental sounds, emotion recognition

- **Zero-Shot Audio Classifier** - Classify audio without predefined categories
  - Flexible classification with custom labels
  - Use cases: Dynamic audio categorization, sound identification

#### Automatic Speech Recognition
- **Whisper** - Convert speech to text with multilingual support
  - Supports 100+ languages
  - Translation mode (translate any language to English)
  - Timestamp options (word-level or sentence-level)
  - Multiple model sizes (tiny to large-v3)
  - Recommended models:
    - `openai/whisper-large-v3` - Best accuracy
    - `openai/whisper-large-v3-turbo` - Fast inference
    - `openai/whisper-small` - Lightweight option
  - Use cases: Transcription, translation, subtitle generation, voice interfaces

- **ChunksToSRT** - Convert transcription chunks to SRT subtitle format
  - Automatic timestamp formatting
  - Time offset support
  - Use cases: Video subtitling, accessibility features

#### Audio Generation
- **Text-to-Speech** - Generate natural-sounding speech from text
  - Multiple voice options
  - Configurable speaking rate and pitch
  - Use cases: Voiceovers, accessibility, content creation

- **Text-to-Audio** - Generate audio effects and sounds from text descriptions
  - Creative sound generation
  - Use cases: Sound effects, audio design, music production

### 📝 Text Processing

#### Text Generation
- **Text Generation** - Generate text using large language models
  - Streaming output support
  - Extensive model support including:
    - Qwen3 series (0.6B to 32B parameters)
    - Meta Llama 3.1 series
    - Ministral 3 series
    - Gemma 3 series
    - TinyLlama for lightweight deployment
  - Quantized model support (BitsAndBytes 4-bit)
  - Configurable parameters:
    - Temperature (0.0-2.0) - Controls randomness
    - Top-p (0.0-1.0) - Controls diversity
    - Max tokens (up to 512 default)
  - GGUF model support for efficient inference
  - Use cases: Chatbots, content generation, code completion, creative writing

#### Text Analysis
- **Text Classification** - Classify text into categories
  - Sentiment analysis
  - Topic categorization
  - Use cases: Content moderation, sentiment analysis, document organization

- **Token Classification** - Identify and classify tokens in text
  - Named entity recognition (NER)
  - Part-of-speech tagging
  - Use cases: Information extraction, text analysis

- **Fill Mask** - Predict masked tokens in text
  - BERT-style masked language modeling
  - Use cases: Text completion, grammar correction

#### Question Answering
- **Question Answering** - Extract answers from context
  - Recommended models:
    - `distilbert-base-cased-distilled-squad`
    - `bert-large-uncased-whole-word-masking-finetuned-squad`
  - Returns answer with confidence score and position
  - Use cases: Document Q&A, customer support, information retrieval

- **Table Question Answering** - Query tabular data with natural language
  - Works with DataFrames
  - Recommended models:
    - `google/tapas-base-finetuned-wtq`
    - `microsoft/tapex-large-finetuned-tabfact`
  - Use cases: Database queries, spreadsheet analysis

#### Text Transformation
- **Translation** - Translate text between languages
  - Multiple language pairs
  - Use cases: Localization, multilingual content

- **Summarization** - Generate concise summaries of long text
  - Extractive and abstractive summarization
  - Use cases: Document summarization, news digests

### 🖼️ Image Analysis

#### Image Classification
- **Image Classifier** - Classify images into predefined categories
  - Recommended models:
    - `google/vit-base-patch16-224` - Vision Transformer
    - `microsoft/resnet-50` - ResNet architecture
    - `Falconsai/nsfw_image_detection` - Content moderation
    - `nateraw/vit-age-classifier` - Age estimation
  - Returns confidence scores for each category
  - Use cases: Content moderation, photo organization, age detection

- **Zero-Shot Image Classifier** - Classify images without training data
  - Uses CLIP models for flexible classification
  - Custom candidate labels
  - Recommended models:
    - `openai/clip-vit-base-patch32`
    - `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
  - Use cases: Dynamic categorization, custom tagging

#### Image Understanding
- **Image Segmentation** - Segment images into different regions
  - Instance and semantic segmentation
  - Use cases: Object isolation, background removal

- **Object Detection** - Detect and locate objects in images
  - Bounding box outputs
  - Multi-object detection
  - Use cases: Surveillance, counting, automation

- **Depth Estimation** - Estimate depth from 2D images
  - Monocular depth prediction
  - Use cases: 3D reconstruction, AR/VR, robotics

### 🎭 Multimodal Processing

#### Video Generation
- **Text-to-Video (CogVideoX)** - Generate videos from text prompts
  - Large diffusion transformer model
  - High-quality, consistent video generation
  - Longer video sequences
  - Use cases: Video content creation, animated storytelling, marketing videos, cinematic content

- **Image-to-Video** - Convert static images into video sequences
  - Animate still images
  - Add motion to photographs
  - Use cases: Photo animation, creating video from stills, dynamic presentations

#### Image-Text Models
- **Image to Text** - Generate captions for images
  - Automatic image captioning
  - Use cases: Accessibility, content tagging, image search

- **Image-Text-to-Text** - Process images with text queries
  - Visual question answering
  - Image reasoning with text context
  - Use cases: Document understanding, visual Q&A, scene description

- **Multimodal** - Process both image and text inputs
  - Vision-language models
  - Combined visual and textual understanding
  - Use cases: Complex visual reasoning, document analysis, multimodal search

### 🎯 Model Customization

#### LoRA (Low-Rank Adaptation)
- **LoRA Selector** - Apply LoRA models to Stable Diffusion
  - Combine up to 5 LoRA models
  - Adjustable strength per LoRA (0.0-2.0)
  - 60+ pre-configured style LoRAs including:
    - Art styles (anime, pixel art, 3D render)
    - Character styles (Ghibli, Arcane, One Piece)
    - Visual effects (fire, lightning, water)
  - Use cases: Style customization, character consistency, artistic effects

- **LoRA Selector XL** - Apply LoRA models to Stable Diffusion XL
  - SDXL-specific LoRA support
  - Enhanced quality for high-resolution outputs
  - Use cases: High-quality style transfer, professional artwork

### 🔧 Utility Nodes

#### Feature Extraction
- **Feature Extraction** - Extract embeddings from text or images
  - Generate vector representations
  - Use cases: Semantic search, similarity matching, clustering

#### Sentence Similarity
- **Sentence Similarity** - Compute similarity between text pairs
  - Use cases: Duplicate detection, semantic search

#### Ranking
- **Ranking** - Rank documents by relevance
  - Use cases: Search engines, recommendation systems

## Installation

```bash
pip install nodetool-huggingface
```

Or install from source:

```bash
git clone https://github.com/nodetool-ai/nodetool-huggingface.git
cd nodetool-huggingface
pip install -e .
```

## Requirements

- Python 3.10+
- PyTorch 2.9.0+
- CUDA support recommended for optimal performance
- See pyproject.toml for full dependencies

## Usage Examples

### Example 1: Text Generation Workflow
```python
from nodetool.nodes.huggingface.text_generation import TextGeneration
from nodetool.workflows.processing_context import ProcessingContext

# Create a text generation node
text_gen = TextGeneration(
    model=HFTextGeneration(repo_id="Qwen/Qwen2.5-7B-Instruct"),
    prompt="Write a short story about a robot learning to paint",
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9
)

# Process in your workflow
result = await text_gen.process(context)
print(result)  # Generated text
```

### Example 2: Image Generation with Stable Diffusion
```python
from nodetool.nodes.huggingface.text_to_image import StableDiffusion

# Create an image generation node
sd = StableDiffusion(
    prompt="A serene landscape with mountains and a lake at sunset, highly detailed",
    negative_prompt="blurry, low quality, distorted",
    width=512,
    height=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42
)

# Generate image
output = await sd.process(context)
# output['image'] contains the generated ImageRef
```

### Example 3: Speech-to-Text Transcription
```python
from nodetool.nodes.huggingface.automatic_speech_recognition import Whisper

# Create a Whisper transcription node
whisper = Whisper(
    model=HFAutomaticSpeechRecognition(repo_id="openai/whisper-large-v3"),
    audio=audio_input,
    task=Task.TRANSCRIBE,
    language=WhisperLanguage.ENGLISH,
    timestamps=Timestamps.WORD
)

# Transcribe audio
result = await whisper.process(context)
print(result['text'])  # Transcribed text
print(result['chunks'])  # Word-level timestamps
```

### Example 4: Image Classification
```python
from nodetool.nodes.huggingface.image_classification import ImageClassifier

# Create an image classifier node
classifier = ImageClassifier(
    model=HFImageClassification(repo_id="google/vit-base-patch16-224"),
    image=image_input
)

# Classify image
results = await classifier.process(context)
# Returns dict of {label: confidence_score}
```

### Example 5: Combining Multiple Nodes in a Workflow
Here's an example of a complete workflow that transcribes audio, generates a summary, and creates an image:

```python
# Step 1: Transcribe audio
transcription = await whisper_node.process(context)

# Step 2: Summarize the transcription
summary_node = TextGeneration(
    prompt=f"Summarize the following text in 2-3 sentences: {transcription['text']}",
    max_new_tokens=256
)
summary = await summary_node.process(context)

# Step 3: Generate an image based on the summary
image_node = StableDiffusion(
    prompt=f"Create an illustration for: {summary}",
    width=768,
    height=512
)
image = await image_node.process(context)
```

## Key Features

### Model Support
- **25+ Node Types**: Comprehensive coverage of HuggingFace model types
- **Streaming Output**: Real-time generation for text and audio
- **Quantization**: Memory-efficient inference with Nunchaku (FP4, INT4)
- **GPU Optimization**: Automatic device management and VRAM optimization
- **CPU Offload**: Run large models on limited hardware
- **LoRA Support**: Easy style customization for Stable Diffusion

### Advanced Capabilities
- **Multimodal Processing**: Combine text, image, and audio in workflows
- **Batch Processing**: Process multiple inputs efficiently
- **Custom Models**: Use any HuggingFace model repo
- **Fine-tuning Ready**: Support for custom LoRA models
- **Recommended Models**: Curated model lists for each node type
- **Flexible Parameters**: Full control over generation parameters

### Developer-Friendly
- **Type Safety**: Full Pydantic type validation
- **Error Handling**: Comprehensive error messages
- **Progress Tracking**: Real-time progress updates for long operations
- **Memory Management**: Automatic cleanup and optimization
- **Documentation**: Detailed docstrings and use cases for all nodes

## Available Workflow Examples

The package includes several pre-built workflow examples that demonstrate how to use the nodes:

- **Image to Image** - Transform images using Stable Diffusion
- **Movie Posters** - Generate movie poster-style images
- **Transcribe Audio** - Convert speech to text with Whisper
- **Pokemon Maker** - Generate Pokemon-style creatures
- **Depth Estimation** - Extract depth information from images
- **Add Subtitles To Video** - Automatically generate and add subtitles
- **Object Detection** - Detect and locate objects in images
- **Summarize Audio** - Transcribe and summarize audio content
- **Segmentation** - Segment images into regions
- **Audio To Spectrogram** - Visualize audio as spectrograms

These examples are located in `src/nodetool/examples/nodetool-huggingface/` and can be imported directly into Nodetool.

## Model Downloads

Models are automatically downloaded from HuggingFace Hub on first use. For better performance:

1. Set your `HF_TOKEN` environment variable for gated models
2. Use `huggingface-cli login` to authenticate
3. Models are cached in `~/.cache/huggingface/` by default
4. Use `allow_patterns` to download only necessary files

### Gated Models
Some models (like FLUX) require accepting terms on HuggingFace:
1. Visit the model page on HuggingFace
2. Accept the terms of use
3. Set your `HF_TOKEN` in Nodetool settings

## Performance Tips

### Memory Optimization
- Use quantized models (INT4, FP4) for reduced VRAM usage
- Enable CPU offload for large models
- Use smaller model variants when possible
- Enable attention slicing for memory-intensive operations

### Speed Optimization
- Use CUDA/GPU when available
- Select appropriate model sizes (tiny/small vs large)
- Use optimized models (e.g., whisper-large-v3-turbo)
- Enable PyTorch 2 attention (automatic)

### Quality vs Performance Trade-offs
- **Fast + Low Memory**: Quantized models with CPU offload
- **Balanced**: FP16 models on GPU
- **Best Quality**: Full precision models with high inference steps

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Enable CPU offload in advanced node properties
- Use quantized models (INT4/FP4)
- Reduce image size or inference steps
- Close other GPU applications

**Model Not Found**
- Ensure model is downloaded first
- Check HuggingFace Hub for model availability
- Verify `HF_TOKEN` is set for gated models

**Slow Inference**
- Check if CUDA is available and being used
- Use smaller or quantized models
- Enable attention optimizations
- Consider using turbo/fast variants

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
2. Inherit from `HuggingFacePipelineNode` or `BaseNode`
3. Implement `preload_model()` and `process()` methods
4. Add docstrings with use cases
5. Include recommended models

## Links & Resources

- [Nodetool Documentation](https://docs.nodetool.ai)
- [HuggingFace Hub](https://huggingface.co/models)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [GitHub Repository](https://github.com/nodetool-ai/nodetool-huggingface)
- [Issue Tracker](https://github.com/nodetool-ai/nodetool-huggingface/issues)
