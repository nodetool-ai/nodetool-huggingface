"""
Smoke-test the local Hugging Face dependency stack used by nodetool-huggingface.

This script is intentionally lightweight:
- validates installed versions for the core HF stack
- imports representative nodetool-huggingface modules
- probes the Mistral tokenizer integration that previously broke Whisper
- optionally prints the exact latest-version pin set from requirements/hf-latest.txt

Usage:
    python scripts/hf_dependency_smoke_test.py
    python scripts/hf_dependency_smoke_test.py --strict
    python scripts/hf_dependency_smoke_test.py --show-latest-pins
"""

from __future__ import annotations

import argparse
import importlib
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT.parent / "nodetool-core" / "src"
HF_SRC = ROOT / "src"
LATEST_PINS = ROOT / "requirements" / "hf-latest.txt"

REQUIRED_PACKAGES: dict[str, str] = {
    "transformers": "5.6.0",
    "diffusers": "0.35.1",
    "huggingface_hub": "1.11.0",
    "accelerate": "1.13.0",
    "safetensors": "0.7.0",
    "tokenizers": "0.22.2",
    "mistral-common": "1.10.0",
    "peft": "0.19.1",
}

MODULE_SMOKES: list[str] = [
    "nodetool.nodes.huggingface.automatic_speech_recognition",
    "nodetool.nodes.huggingface.text_generation",
    "nodetool.nodes.huggingface.image_text_to_text",
    "nodetool.nodes.huggingface.text_to_image",
]


def normalize_version(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in value.replace("-", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def version_ok(installed: str, minimum: str) -> bool:
    return normalize_version(installed) >= normalize_version(minimum)


def iter_latest_pins() -> Iterable[str]:
    if not LATEST_PINS.exists():
        return []
    return [
        line.strip()
        for line in LATEST_PINS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def print_versions(strict: bool) -> int:
    failures = 0
    print("== Installed HF stack ==")
    for package, minimum in REQUIRED_PACKAGES.items():
        try:
            installed = metadata.version(package)
            ok = version_ok(installed, minimum)
            status = "OK" if ok else "OLD"
            print(f"[{status}] {package} {installed} (min {minimum})")
            if strict and not ok:
                failures += 1
        except metadata.PackageNotFoundError:
            print(f"[MISS] {package} not installed")
            failures += 1
    return failures


def smoke_imports() -> int:
    failures = 0
    sys.path.insert(0, str(CORE_SRC))
    sys.path.insert(0, str(HF_SRC))

    print("\n== nodetool-huggingface module imports ==")
    for module_name in MODULE_SMOKES:
        try:
            importlib.import_module(module_name)
            print(f"[OK]   {module_name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {module_name}: {exc}")

    print("\n== Mistral tokenizer compatibility ==")
    try:
        module = importlib.import_module("mistral_common.protocol.instruct.request")
        missing = [name for name in ["ChatCompletionRequest", "ReasoningEffort"] if not hasattr(module, name)]
        if missing:
            failures += 1
            print(f"[FAIL] mistral_common.protocol.instruct.request missing: {', '.join(missing)}")
        else:
            print("[OK]   mistral_common.protocol.instruct.request exports expected symbols")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"[FAIL] mistral_common.protocol.instruct.request import: {exc}")

    try:
        module = importlib.import_module("mistral_common.tokens.tokenizers.utils")
        missing = [name for name in ["get_one_valid_tokenizer_file"] if not hasattr(module, name)]
        if missing:
            failures += 1
            print(f"[FAIL] mistral_common.tokens.tokenizers.utils missing: {', '.join(missing)}")
        else:
            print("[OK]   mistral_common.tokens.tokenizers.utils exports expected symbols")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"[FAIL] mistral_common.tokens.tokenizers.utils import: {exc}")

    try:
        importlib.import_module("transformers.tokenization_mistral_common")
        print("[OK]   transformers.tokenization_mistral_common")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"[FAIL] transformers.tokenization_mistral_common: {exc}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test Hugging Face dependency compatibility.")
    parser.add_argument("--strict", action="store_true", help="Fail if installed versions are below the recommended minimums.")
    parser.add_argument("--show-latest-pins", action="store_true", help="Print the exact latest pin set from requirements/hf-latest.txt.")
    args = parser.parse_args()

    failures = 0
    failures += print_versions(strict=args.strict)
    failures += smoke_imports()

    if args.show_latest_pins:
        print("\n== Latest pin set ==")
        for line in iter_latest_pins():
            print(line)

    print(f"\nDone. failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
