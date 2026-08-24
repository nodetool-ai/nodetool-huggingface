"""Regression: DanceDiffusion stopped loading its own checkpoint on diffusers 0.36+.

``huggingface.text_to_audio.DanceDiffusion`` loads ``harmonai/maestro-150k``,
published against diffusers 0.5. On a RunPod 4xA40 (diffusers 0.40.0,
transformers 5.15.1, torchaudio 2.9.0+cu128, pyannote.audio 3.4.0, Python 3.11)
the run died with::

    Cannot load  because time_proj.weight expected shape torch.Size([128]),
    but got torch.Size([8]). If you want to instead overwrite randomly
    initialized weights, please make sure to pass both
    `low_cpu_mem_usage=False` and `ignore_mismatched_sizes=True`

diffusers 0.36 widened ``UNet1DModel``'s Fourier time embedding from a
hardcoded ``embedding_size=8`` to ``(time_embedding_dim or
block_out_channels[0] * 2) // 2`` -- 128 for these repos. Passing
``time_embedding_dim=16`` puts it back at 8, and every checkpoint tensor then
matches.

The error message's own suggestion is wrong here, and the node must not take
it: with ``ignore_mismatched_sizes=True`` the Fourier embedding is 256 wide
instead of 16, the first block receives 258 channels where it wants
``in_channels + extra_in_channels == 18``, and the run dies inside
``F.conv1d``. Measured on diffusers 0.38.0 / torch 2.9.0 against the real
checkpoint::

    RuntimeError: Given groups=1, weight of size [128, 18, 1], expected
    input[1, 258, 65536] to have 18 channels, but got 258 channels instead

No network and no download here: the checkpoint's own ``unet/config.json`` is
inlined, and ``_FakeUNet1DModel`` reproduces diffusers 0.36+'s width rule and
its shape check.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT.parent / "nodetool-core" / "src"
HF_SRC = ROOT / "src"
for _p in (str(CORE_SRC), str(HF_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nodetool.nodes.huggingface.text_to_audio import DanceDiffusion  # noqa: E402

# Verbatim https://huggingface.co/harmonai/maestro-150k/resolve/main/unet/config.json
MAESTRO_UNET_CONFIG: dict[str, Any] = {
    "_class_name": "UNet1DModel",
    "_diffusers_version": "0.5.0.dev0",
    "block_out_channels": [
        128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512,
    ],
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 2,
    "mid_block_type": "UNetMidBlock1D",
    "extra_in_channels": 16,
    "out_channels": 2,
    "sample_rate": 16000,
    "sample_size": 65536,
    "time_embedding_type": "fourier",
    "use_timestep_embedding": False,
}

# What harmonai actually stores for time_proj.weight.
CHECKPOINT_TIME_PROJ_WIDTH = 8


class _FakeUNet1DModel:
    """diffusers 0.36+'s Fourier width rule and its shape check, nothing else."""

    # The signature matters: the node probes it to find out whether the
    # installed diffusers is one of the versions that widened the embedding.
    def __init__(
        self,
        time_embedding_type: str = "fourier",
        time_embedding_dim: int | None = None,
        **_,
    ):
        block_out_channels = MAESTRO_UNET_CONFIG["block_out_channels"]
        self.time_proj_width = (
            time_embedding_dim or block_out_channels[0] * 2
        ) // 2

    @classmethod
    def load_config(cls, repo_id: str, subfolder: str | None = None, **_):
        assert subfolder == "unet"
        return dict(MAESTRO_UNET_CONFIG)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        subfolder: str | None = None,
        time_embedding_dim: int | None = None,
        ignore_mismatched_sizes: bool = False,
        **_,
    ):
        model = cls(time_embedding_dim=time_embedding_dim)
        if (
            model.time_proj_width != CHECKPOINT_TIME_PROJ_WIDTH
            and not ignore_mismatched_sizes
        ):
            raise ValueError(
                f"Cannot load  because time_proj.weight expected shape "
                f"torch.Size([{model.time_proj_width}]), but got "
                f"torch.Size([{CHECKPOINT_TIME_PROJ_WIDTH}])."
            )
        return model


class _FakeDiffusionPipeline:
    """Builds its own UNet when the caller does not hand one over."""

    def __init__(self, unet):
        self.unet = unet

    @classmethod
    def from_pretrained(cls, repo_id: str, unet=None, **_):
        if unet is None:
            unet = _FakeUNet1DModel.from_pretrained(repo_id, subfolder="unet")
        return cls(unet)


@pytest.fixture
def fake_diffusers(monkeypatch):
    import diffusers
    import diffusers.pipelines.pipeline_utils as pipeline_utils

    monkeypatch.setattr(diffusers, "UNet1DModel", _FakeUNet1DModel)
    monkeypatch.setattr(pipeline_utils, "DiffusionPipeline", _FakeDiffusionPipeline)
    return _FakeUNet1DModel


@pytest.mark.asyncio
async def test_preload_loads_the_checkpoint_unmodified(monkeypatch, fake_diffusers):
    """The whole defect: preload_model must not raise, and must not randomize."""
    loaded: list[dict[str, Any]] = []

    async def fake_load_model(self, context, model_class, model_id, **kwargs):
        loaded.append({"model_class": model_class, "kwargs": kwargs})
        return model_class.from_pretrained(model_id, **kwargs)

    monkeypatch.setattr(DanceDiffusion, "load_model", fake_load_model)

    node = DanceDiffusion(id="dance")
    await node.preload_model(context=None)  # type: ignore[arg-type]

    unet = node._pipeline.unet
    assert unet.time_proj_width == CHECKPOINT_TIME_PROJ_WIDTH

    # Randomizing time_proj.weight throws away the trained Fourier basis and
    # does not even produce a runnable model. Never take that escape hatch.
    assert all(
        not call["kwargs"].get("ignore_mismatched_sizes") for call in loaded
    )


def test_legacy_fourier_checkpoint_gets_the_pre_036_width():
    from nodetool.nodes.huggingface.text_to_audio import (
        _legacy_fourier_time_embedding_dim,
    )

    assert _legacy_fourier_time_embedding_dim(MAESTRO_UNET_CONFIG) == 16


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"time_embedding_dim": 256}, id="checkpoint-states-its-width"),
        pytest.param({"time_embedding_type": "positional"}, id="not-fourier"),
        pytest.param({"use_timestep_embedding": True}, id="feeds-time-mlp"),
    ],
)
def test_configs_the_override_must_not_touch(override):
    from nodetool.nodes.huggingface.text_to_audio import (
        _legacy_fourier_time_embedding_dim,
    )

    config = {**MAESTRO_UNET_CONFIG, **override}
    assert _legacy_fourier_time_embedding_dim(config) is None
