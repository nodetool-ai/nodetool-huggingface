import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
from nodetool.nodes.huggingface.text_to_image import Flux, FluxVariant, FluxQuantization
from tests.test_utils import main


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    flux = Flux()
    flux.variant = FluxVariant.DEV
    flux.quantization = FluxQuantization.NUNCHAKU_4BIT
    flux.height = 512
    flux.width = 512
    flux.num_inference_steps = 4

    main([flux])
