import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
from nodetool.dsl.huggingface.audio_classification import ZeroShotAudioClassifier
from tests.test_utils import main


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    node = ZeroShotAudioClassifier()
    node.model.repo_id = "laion/clap-htsat-unfused"
    node.__class__.requires_gpu = lambda self: False  # type: ignore[assignment]
    main([node])
