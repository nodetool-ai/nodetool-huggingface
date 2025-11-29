import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
from nodetool.dsl.huggingface.text_to_image import QwenImage
from tests.test_utils import main


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    qwen_image = QwenImage()
    qwen_image.model.repo_id = "nunchaku-tech/nunchaku-qwen-image"
    qwen_image.model.path = "svdq-int4_r32-qwen-image.safetensors"
    qwen_image.height = 512
    qwen_image.width = 512
    qwen_image.num_inference_steps = 4
    qwen_image.enable_cpu_offload = True

    main([qwen_image])
