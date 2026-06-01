import torch

from nodetool.nodes.huggingface.text_to_image import Bria
from nodetool.metadata.types import HFTextToImage


def test_bria_node_metadata():
    bria = Bria()

    # Node identity and UI metadata
    assert bria.get_node_type() == "huggingface.text_to_image.Bria"
    assert Bria.get_title() == "Bria 3.2"
    assert bria.get_model_id() == "briaai/BRIA-3.2"
    assert Bria.get_basic_fields() == ["prompt", "height", "width", "seed"]

    # Bria-appropriate defaults (match the diffusers BriaPipeline signature)
    assert bria.guidance_scale == 5.0
    assert bria.num_inference_steps == 30
    assert bria.max_sequence_length == 128

    # Recommended model points at the gated BRIA-3.2 repo
    models = Bria.get_recommended_models()
    assert len(models) == 1
    assert isinstance(models[0], HFTextToImage)
    assert models[0].repo_id == "briaai/BRIA-3.2"


class _Recorder:
    """Records the dtype it was last cast to via ``.to(dtype=...)``."""

    def __init__(self):
        self.dtype = None

    def to(self, *args, dtype=None, **kwargs):
        if dtype is not None:
            self.dtype = dtype
        return self


class _DenseReluDense:
    def __init__(self):
        self.wo = _Recorder()


class _BlockLayer:
    def __init__(self):
        self.DenseReluDense = _DenseReluDense()


class _Block:
    def __init__(self):
        # The pipeline only touches the final layer (``layer[-1]``).
        self.layer = [_BlockLayer(), _BlockLayer()]


class _Encoder:
    def __init__(self, num_blocks=2):
        self.block = [_Block() for _ in range(num_blocks)]


class _TextEncoder(_Recorder):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()


class _VaeConfig:
    def __init__(self, shift_factor):
        self.shift_factor = shift_factor


class _Vae(_Recorder):
    def __init__(self, shift_factor):
        super().__init__()
        self.config = _VaeConfig(shift_factor)


class _MockPipeline:
    def __init__(self, vae_shift_factor=0):
        self.text_encoder = _TextEncoder()
        self.vae = _Vae(vae_shift_factor)
        self.moved_to = None
        self.cpu_offload_enabled = False
        self.attention_slicing_enabled = False

    def to(self, device):
        self.moved_to = device
        return self

    def enable_model_cpu_offload(self):
        self.cpu_offload_enabled = True

    def enable_attention_slicing(self):
        self.attention_slicing_enabled = True


def test_bria_precision_fixes_with_zero_shift_factor():
    bria = Bria()
    pipeline = _MockPipeline(vae_shift_factor=0)
    bria._pipeline = pipeline

    bria._apply_precision_fixes()

    # T5 encoder cast to bfloat16, but the final dense layer kept in float32.
    assert pipeline.text_encoder.dtype == torch.bfloat16
    for block in pipeline.text_encoder.encoder.block:
        assert block.layer[-1].DenseReluDense.wo.dtype == torch.float32

    # VAE forced to float32 when shift_factor == 0.
    assert pipeline.vae.dtype == torch.float32


def test_bria_precision_fixes_leaves_vae_untouched_when_shift_factor_nonzero():
    bria = Bria()
    pipeline = _MockPipeline(vae_shift_factor=0.5)
    bria._pipeline = pipeline

    bria._apply_precision_fixes()

    assert pipeline.text_encoder.dtype == torch.bfloat16
    assert pipeline.vae.dtype is None


async def test_bria_move_to_device_cpu():
    bria = Bria()
    pipeline = _MockPipeline()
    bria._pipeline = pipeline

    await bria.move_to_device("cpu")

    # Moving to CPU should physically relocate the pipeline (offload disabled).
    assert pipeline.moved_to == "cpu"
    # Attention slicing is skipped on CPU.
    assert pipeline.attention_slicing_enabled is False


async def test_bria_move_to_device_cuda_uses_cpu_offload():
    bria = Bria()
    bria.enable_cpu_offload = True
    bria.enable_attention_slicing = True
    pipeline = _MockPipeline()
    bria._pipeline = pipeline

    await bria.move_to_device("cuda")

    # With CPU offload enabled, the pipeline is not moved wholesale to the GPU.
    assert pipeline.cpu_offload_enabled is True
    assert pipeline.moved_to is None
    assert pipeline.attention_slicing_enabled is True


if __name__ == "__main__":
    import asyncio

    test_bria_node_metadata()
    test_bria_precision_fixes_with_zero_shift_factor()
    test_bria_precision_fixes_leaves_vae_untouched_when_shift_factor_nonzero()
    asyncio.run(test_bria_move_to_device_cpu())
    asyncio.run(test_bria_move_to_device_cuda_uses_cpu_offload())
    print("All Bria smoke tests passed")
