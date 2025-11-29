import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.dsl.huggingface.text_to_audio import DanceDiffusion
from tests.test_utils import main


if __name__ == "__main__":
    main([
        DanceDiffusion(),
    ])
