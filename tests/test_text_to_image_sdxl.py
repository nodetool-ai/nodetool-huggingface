import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import nodetool.metadata.types as types
from nodetool.dsl.huggingface.text_to_image import StableDiffusionXL
from tests.test_utils import main


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    sdxl = StableDiffusionXL()
    sdxl.model = types.HFStableDiffusionXL(
        type="hf.stable_diffusion_xl",
        repo_id="nunchaku-tech/nunchaku-sdxl",
        path="svdq-int4_r32-sdxl.safetensors",
        variant=None,
        allow_patterns=None,
        ignore_patterns=None,
    )
    sdxl.height = 512
    sdxl.width = 512
    sdxl.num_inference_steps = 4
    sdxl.enable_cpu_offload = True

    main([sdxl])
