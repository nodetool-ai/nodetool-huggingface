import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import io
import os
import nodetool.metadata.types as types
from PIL import Image
from nodetool.dsl.huggingface.text_to_image import FluxControl
from tests.test_utils import main


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    flux_control = FluxControl()
    flux_control.model = types.HFControlNetFlux(
        type="hf.controlnet_flux",
        repo_id="nunchaku-tech/nunchaku-flux.1-depth-dev",
        path="svdq-int4_r32-flux.1-depth-dev.safetensors",
        variant=None,
        allow_patterns=None,
        ignore_patterns=None,
    )
    flux_control.height = 512
    flux_control.width = 512
    flux_control.num_inference_steps = 4

    blank = Image.new("RGB", (512, 512), color=(0, 0, 0))
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    flux_control.control_image = types.ImageRef(
        type="image",
        uri="",
        asset_id=None,
        data=buf.getvalue(),
    )

    main([flux_control])
