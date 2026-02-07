"""
Test that all node modules can be imported successfully.
"""
import pytest


@pytest.mark.unit
def test_import_audio_classification():
    """Test that audio classification nodes can be imported."""
    from nodetool.nodes.huggingface import audio_classification
    assert hasattr(audio_classification, "AudioClassifier")
    assert hasattr(audio_classification, "ZeroShotAudioClassifier")


@pytest.mark.unit
def test_import_automatic_speech_recognition():
    """Test that ASR nodes can be imported."""
    from nodetool.nodes.huggingface import automatic_speech_recognition
    assert hasattr(automatic_speech_recognition, "Whisper")
    assert hasattr(automatic_speech_recognition, "ChunksToSRT")


@pytest.mark.unit
def test_import_image_classification():
    """Test that image classification nodes can be imported."""
    from nodetool.nodes.huggingface import image_classification
    assert hasattr(image_classification, "ImageClassifier")
    assert hasattr(image_classification, "ZeroShotImageClassifier")


@pytest.mark.unit
def test_import_image_segmentation():
    """Test that image segmentation nodes can be imported."""
    from nodetool.nodes.huggingface import image_segmentation
    assert hasattr(image_segmentation, "FindSegment")
    assert hasattr(image_segmentation, "Segmentation")


@pytest.mark.unit
def test_import_object_detection():
    """Test that object detection nodes can be imported."""
    from nodetool.nodes.huggingface import object_detection
    assert hasattr(object_detection, "ObjectDetection")
    assert hasattr(object_detection, "ZeroShotObjectDetection")


@pytest.mark.unit
def test_import_text_generation():
    """Test that text generation nodes can be imported."""
    from nodetool.nodes.huggingface import text_generation
    assert hasattr(text_generation, "TextGeneration")


@pytest.mark.unit
def test_import_text_classification():
    """Test that text classification nodes can be imported."""
    from nodetool.nodes.huggingface import text_classification
    assert hasattr(text_classification, "TextClassifier")
    assert hasattr(text_classification, "ZeroShotTextClassifier")


@pytest.mark.unit
def test_import_question_answering():
    """Test that question answering nodes can be imported."""
    from nodetool.nodes.huggingface import question_answering
    assert hasattr(question_answering, "QuestionAnswering")
    assert hasattr(question_answering, "TableQuestionAnswering")


@pytest.mark.unit
def test_import_image_to_text():
    """Test that image to text nodes can be imported."""
    from nodetool.nodes.huggingface import image_to_text
    assert hasattr(image_to_text, "ImageToText")


@pytest.mark.unit
def test_import_depth_estimation():
    """Test that depth estimation nodes can be imported."""
    from nodetool.nodes.huggingface import depth_estimation
    assert hasattr(depth_estimation, "DepthEstimation")


@pytest.mark.unit
def test_import_feature_extraction():
    """Test that feature extraction nodes can be imported."""
    from nodetool.nodes.huggingface import feature_extraction
    assert hasattr(feature_extraction, "FeatureExtraction")


@pytest.mark.unit
def test_import_fill_mask():
    """Test that fill mask nodes can be imported."""
    from nodetool.nodes.huggingface import fill_mask
    assert hasattr(fill_mask, "FillMask")


@pytest.mark.unit
def test_import_translation():
    """Test that translation nodes can be imported."""
    from nodetool.nodes.huggingface import translation
    assert hasattr(translation, "Translation")


@pytest.mark.unit
def test_import_text_to_speech():
    """Test that text to speech nodes can be imported."""
    from nodetool.nodes.huggingface import text_to_speech
    assert hasattr(text_to_speech, "Bark")


@pytest.mark.unit
def test_import_lora():
    """Test that LoRA nodes can be imported."""
    from nodetool.nodes.huggingface import lora
    assert hasattr(lora, "LoRASelector")
    assert hasattr(lora, "LoRASelectorXL")
