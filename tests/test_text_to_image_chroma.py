import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
from nodetool.dsl.huggingface.text_to_image import Chroma
from tests.test_utils import main


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    chroma = Chroma()
    chroma.height = 512
    chroma.width = 512
    chroma.num_inference_steps = 4

    main([chroma])
