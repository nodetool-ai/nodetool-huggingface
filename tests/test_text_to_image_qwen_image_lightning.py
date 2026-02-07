import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
from nodetool.dsl.huggingface.text_to_image import QwenImageLightning
from tests.test_utils import main


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    qwen_lightning = QwenImageLightning()
    qwen_lightning.model.repo_id = "nunchaku-ai/nunchaku-qwen-image"
    qwen_lightning.model.path = "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"
    qwen_lightning.height = 512
    qwen_lightning.width = 512
    qwen_lightning.num_inference_steps = 4
    qwen_lightning.rank = 32

    main([qwen_lightning])
