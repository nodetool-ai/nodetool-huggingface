import pytest
from nodetool.nodes.huggingface.text_to_image import QwenImage, QwenQuantization
from tests.test_utils import main

async def test_qwen_image_quantization():
    qwen = QwenImage()
    qwen.quantization = QwenQuantization.INT4

    # Verify that the model ID and path are resolved correctly
    model_config = qwen._resolve_model_config()
    assert model_config.repo_id == "nunchaku-tech/nunchaku-qwen-image"
    assert "int4" in model_config.path

    qwen.quantization = QwenQuantization.FP4
    model_config = qwen._resolve_model_config()
    assert model_config.repo_id == "nunchaku-tech/nunchaku-qwen-image"
    assert "fp4" in model_config.path

    qwen.quantization = QwenQuantization.FP16
    model_config = qwen._resolve_model_config()
    assert model_config.repo_id == "Qwen/Qwen-Image"

    # Test move_to_device (mocking pipeline)
    class MockPipeline:
        def __init__(self):
            self.transformer = MockTransformer()
            self.text_encoder = MockTransformer()
            self.vae = MockTransformer()
        def to(self, device):
            self.transformer.to(device)
            self.text_encoder.to(device)
            self.vae.to(device)

    class MockTransformer:
        def to(self, device):
            pass

    qwen._pipeline = MockPipeline()
    await qwen.move_to_device("cpu")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_qwen_image_quantization())
