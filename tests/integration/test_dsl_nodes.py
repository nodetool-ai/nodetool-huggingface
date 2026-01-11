"""
Integration tests for DSL node execution.
These tests verify that nodes can be instantiated and have correct structure.
"""
import pytest


@pytest.mark.integration
def test_audio_classifier_dsl_node():
    """Test AudioClassifier DSL node can be created."""
    from nodetool.dsl.huggingface.audio_classification import AudioClassifier
    
    node = AudioClassifier()
    assert hasattr(node, 'model')
    assert hasattr(node, 'audio')
    
    # Check that it has a repo_id in the model
    assert hasattr(node.model, 'repo_id')


@pytest.mark.integration
def test_image_classifier_dsl_node():
    """Test ImageClassifier DSL node can be created."""
    from nodetool.dsl.huggingface.image_classification import ImageClassifier
    
    node = ImageClassifier()
    assert hasattr(node, 'model')
    assert hasattr(node, 'image')


@pytest.mark.integration
def test_whisper_dsl_node():
    """Test Whisper DSL node can be created."""
    from nodetool.dsl.huggingface.automatic_speech_recognition import Whisper
    
    node = Whisper()
    assert hasattr(node, 'model')
    assert hasattr(node, 'audio')
    assert hasattr(node, 'task')
    assert hasattr(node, 'language')


@pytest.mark.integration
def test_text_generation_dsl_node():
    """Test TextGeneration DSL node can be created."""
    from nodetool.dsl.huggingface.text_generation import TextGeneration
    
    node = TextGeneration()
    assert hasattr(node, 'model')
    assert hasattr(node, 'prompt')
    assert hasattr(node, 'max_new_tokens')
    assert hasattr(node, 'temperature')


@pytest.mark.integration
def test_object_detection_dsl_node():
    """Test ObjectDetection DSL node can be created."""
    from nodetool.dsl.huggingface.object_detection import ObjectDetection
    
    node = ObjectDetection()
    assert hasattr(node, 'model')
    assert hasattr(node, 'image')
    assert hasattr(node, 'threshold')


@pytest.mark.integration
def test_depth_estimation_dsl_node():
    """Test DepthEstimation DSL node can be created."""
    from nodetool.dsl.huggingface.depth_estimation import DepthEstimation
    
    node = DepthEstimation()
    assert hasattr(node, 'model')
    assert hasattr(node, 'image')


@pytest.mark.integration
def test_question_answering_dsl_node():
    """Test QuestionAnswering DSL node can be created."""
    from nodetool.dsl.huggingface.question_answering import QuestionAnswering
    
    node = QuestionAnswering()
    assert hasattr(node, 'model')
    assert hasattr(node, 'question')
    assert hasattr(node, 'context')
