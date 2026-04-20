from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT.parent / "nodetool-core" / "src"
HF_SRC = ROOT / "src"


def test_huggingface_tts_imports_with_current_core_layout() -> None:
    sys.path.insert(0, str(CORE_SRC))
    sys.path.insert(0, str(HF_SRC))

    local_provider_utils = importlib.import_module(
        "nodetool.huggingface.local_provider_utils"
    )
    text_to_speech = importlib.import_module("nodetool.nodes.huggingface.text_to_speech")

    assert hasattr(local_provider_utils, "pipeline_progress_callback")
    assert text_to_speech.KokoroTTS.get_node_type() == "huggingface.text_to_speech.KokoroTTS"


def test_sentence_transformers_node_imports_with_current_core_layout() -> None:
    sys.path.insert(0, str(CORE_SRC))
    sys.path.insert(0, str(HF_SRC))

    sentence_transformers = importlib.import_module(
        "nodetool.nodes.huggingface.sentence_transformers"
    )

    assert (
        sentence_transformers.SplitSentences.get_node_type()
        == "huggingface.sentence_transformers.SplitSentences"
    )
