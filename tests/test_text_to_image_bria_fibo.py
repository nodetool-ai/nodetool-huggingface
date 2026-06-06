import pytest

from nodetool.nodes.huggingface.text_to_image import BriaFibo


def test_bria_fibo_basics():
    node = BriaFibo()

    # Model routing
    assert node.get_model_id() == "briaai/FIBO"
    assert node.get_title() == "Bria FIBO"

    # Recommended model points at the gated FIBO repo
    recommended = BriaFibo.get_recommended_models()
    assert len(recommended) == 1
    assert recommended[0].repo_id == "briaai/FIBO"
    # Custom pipeline code must be downloaded for trust_remote_code to work
    assert any("*.py" in p for p in (recommended[0].allow_patterns or []))

    # Defaults reflect FIBO recommendations
    assert node.guidance_scale == 5.0
    assert node.num_inference_steps == 50
    assert node.max_sequence_length == 3000

    # Default prompt is structured JSON
    import json

    json.loads(node.prompt)

    assert "prompt" in BriaFibo.get_basic_fields()


@pytest.mark.asyncio
async def test_bria_fibo_move_to_device_cpu_offload():
    node = BriaFibo()

    class MockPipeline:
        def __init__(self):
            self.offloaded = False
            self.moved_to = None

        def to(self, device):
            self.moved_to = device

        def enable_model_cpu_offload(self):
            self.offloaded = True

    node._pipeline = MockPipeline()

    # With CPU offload enabled, moving to cuda should enable offload.
    node.enable_cpu_offload = True
    await node.move_to_device("cuda")
    assert node._pipeline.offloaded is True

    # Moving to cpu should move the whole pipeline to cpu.
    await node.move_to_device("cpu")
    assert node._pipeline.moved_to == "cpu"

    # Without CPU offload, it should move directly to the device.
    node2 = BriaFibo()
    node2.enable_cpu_offload = False
    node2._pipeline = MockPipeline()
    await node2.move_to_device("cuda")
    assert node2._pipeline.moved_to == "cuda"


if __name__ == "__main__":
    import asyncio

    test_bria_fibo_basics()
    asyncio.run(test_bria_fibo_move_to_device_cpu_offload())
