"""
Unit tests for HuggingFace node base classes.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


@pytest.mark.unit
def test_huggingface_pipeline_node_initialization():
    """Test that HuggingFacePipelineNode can be initialized."""
    from nodetool.nodes.huggingface.huggingface_pipeline import HuggingFacePipelineNode
    
    # This is an abstract class, so we can check that it exists
    assert hasattr(HuggingFacePipelineNode, 'preload_model')
    assert hasattr(HuggingFacePipelineNode, 'load_pipeline')


@pytest.mark.unit
def test_audio_classifier_node_creation():
    """Test AudioClassifier node can be created."""
    from nodetool.nodes.huggingface.audio_classification import AudioClassifier
    
    node = AudioClassifier()
    assert hasattr(node, 'model')
    assert hasattr(node, 'audio')
    assert hasattr(node, 'process')


@pytest.mark.unit
def test_image_classifier_node_creation():
    """Test ImageClassifier node can be created."""
    from nodetool.nodes.huggingface.image_classification import ImageClassifier
    
    node = ImageClassifier()
    assert hasattr(node, 'model')
    assert hasattr(node, 'image')
    assert hasattr(node, 'process')


@pytest.mark.unit
def test_text_classifier_node_creation():
    """Test TextClassifier node can be created."""
    from nodetool.nodes.huggingface.text_classification import TextClassifier
    
    node = TextClassifier()
    assert hasattr(node, 'model')
    assert hasattr(node, 'prompt')
    assert hasattr(node, 'process')


@pytest.mark.unit
def test_whisper_node_creation():
    """Test Whisper node can be created."""
    from nodetool.nodes.huggingface.automatic_speech_recognition import Whisper
    
    node = Whisper()
    assert hasattr(node, 'model')
    assert hasattr(node, 'audio')
    assert hasattr(node, 'task')
    assert hasattr(node, 'language')
    assert hasattr(node, 'process')


@pytest.mark.unit
def test_chunks_to_srt_node_creation():
    """Test ChunksToSRT node can be created."""
    from nodetool.nodes.huggingface.automatic_speech_recognition import ChunksToSRT
    from nodetool.metadata.types import AudioChunk
    
    node = ChunksToSRT()
    assert hasattr(node, 'chunks')
    assert hasattr(node, 'time_offset')
    assert hasattr(node, 'process')


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chunks_to_srt_process():
    """Test ChunksToSRT processing logic."""
    from nodetool.nodes.huggingface.automatic_speech_recognition import ChunksToSRT
    from nodetool.metadata.types import AudioChunk
    
    node = ChunksToSRT()
    node.chunks = [
        AudioChunk(timestamp=(0.0, 2.5), text="Hello world"),
        AudioChunk(timestamp=(2.5, 5.0), text="This is a test"),
    ]
    node.time_offset = 0.0
    
    # Mock the context
    mock_context = MagicMock()
    
    result = await node.process(mock_context)
    
    # Check that result is a string containing SRT format
    assert isinstance(result, str)
    assert "1" in result  # Sequence number
    assert "00:00:00,000 --> 00:00:02,500" in result
    assert "Hello world" in result
    assert "2" in result
    assert "This is a test" in result


@pytest.mark.unit
def test_depth_estimation_node_creation():
    """Test DepthEstimation node can be created."""
    from nodetool.nodes.huggingface.depth_estimation import DepthEstimation
    
    node = DepthEstimation()
    assert hasattr(node, 'model')
    assert hasattr(node, 'image')
    assert hasattr(node, 'process')


@pytest.mark.unit
def test_feature_extraction_node_creation():
    """Test FeatureExtraction node can be created."""
    from nodetool.nodes.huggingface.feature_extraction import FeatureExtraction
    
    node = FeatureExtraction()
    assert hasattr(node, 'model')
    assert hasattr(node, 'inputs')
    assert hasattr(node, 'process')


@pytest.mark.unit
def test_fill_mask_node_creation():
    """Test FillMask node can be created."""
    from nodetool.nodes.huggingface.fill_mask import FillMask
    
    node = FillMask()
    assert hasattr(node, 'model')
    assert hasattr(node, 'inputs')
    assert hasattr(node, 'top_k')
    assert hasattr(node, 'process')


@pytest.mark.unit
def test_object_detection_node_creation():
    """Test ObjectDetection node can be created."""
    from nodetool.nodes.huggingface.object_detection import ObjectDetection
    
    node = ObjectDetection()
    assert hasattr(node, 'model')
    assert hasattr(node, 'image')
    assert hasattr(node, 'threshold')
    assert hasattr(node, 'process')


@pytest.mark.unit
def test_question_answering_node_creation():
    """Test QuestionAnswering node can be created."""
    from nodetool.nodes.huggingface.question_answering import QuestionAnswering
    
    node = QuestionAnswering()
    assert hasattr(node, 'model')
    assert hasattr(node, 'question')
    assert hasattr(node, 'context')
    assert hasattr(node, 'process')


@pytest.mark.unit
def test_translation_node_creation():
    """Test Translation node can be created."""
    from nodetool.nodes.huggingface.translation import Translation
    
    node = Translation()
    assert hasattr(node, 'model')
    assert hasattr(node, 'inputs')
    assert hasattr(node, 'process')

