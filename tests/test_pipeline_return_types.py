import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nodetool.nodes.huggingface.huggingface_pipeline import extract_generated_text


def test_extract_from_string_prompt_records():
    """The shape returned for a plain string prompt."""
    result = [{"input_text": "", "generated_text": "a cat on a mat"}]
    assert extract_generated_text(result) == "a cat on a mat"


def test_extract_from_chat_formatted_records():
    """Chat-formatted prompts put the answer in a trailing assistant message."""
    result = [
        {
            "input_text": [{"role": "user", "content": [{"type": "image"}]}],
            "generated_text": [
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "assistant", "content": "a cat on a mat"},
            ],
        }
    ]
    assert extract_generated_text(result) == "a cat on a mat"


def test_extract_from_chat_content_parts():
    result = [
        {
            "generated_text": [
                {"role": "user", "content": [{"type": "text", "text": "describe"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "a cat"},
                        {"type": "text", "text": " on a mat"},
                    ],
                },
            ]
        }
    ]
    assert extract_generated_text(result) == "a cat on a mat"


def test_extract_from_bare_dict():
    assert extract_generated_text({"generated_text": "hello"}) == "hello"


def test_extract_from_bare_string():
    assert extract_generated_text("hello") == "hello"


def test_extract_from_empty_result():
    assert extract_generated_text([]) == ""
    assert extract_generated_text([{"generated_text": []}]) == ""
