#!/usr/bin/env python3
"""
Batch Thumbnail Prompt Generator

Reads all workflow JSON files in the nodetool-huggingace examples directory
and generates image prompts for their thumbnails using an AI agent.

Usage:
    python batch_thumbnail_prompts.py [--output OUTPUT_FILE] [--dry-run]

Options:
    --output    Output JSON file for results (default: thumbnail_prompts.json)
    --dry-run   Show what would be processed without running the agent
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.text import FormatText, LoadTextFolder, ParseJSON, ExtractJSON, Join
from nodetool.dsl.nodetool.control import Collect
from nodetool.dsl.nodetool.output import Output
from nodetool.dsl.nodetool.image import SaveImageFile
from nodetool.dsl.nodetool.text import RegexReplace
from nodetool.dsl.kie.image import NanoBanana
from nodetool.metadata.types import LanguageModel, Provider


# System prompt for the thumbnail generation agent
THUMBNAIL_SYSTEM_PROMPT = """You are a creative designer specialized in creating compelling thumbnail imagery for software workflows.

Your task is to analyze a NodeTool workflow and generate a single image prompt that:
1. Visually represents the workflow's purpose and functionality
2. Is suitable for a 1024x1024 thumbnail image
3. Uses modern design aesthetics
4. Dark color palette
5. Simple shapes and lines
6. No text or labels

Output Format:
Return ONLY the image generation prompt, nothing else. No explanations, no prefixes.
The prompt should be 1-3 sentences, descriptive and visually focused."""


def create_workflow_graph(examples_dir: Path, single_file: str | None = None):
    """Create a NodeTool workflow graph that processes files using LoadTextFolder."""

    # Load JSON files from the examples directory using LoadTextFolder node
    load_text = LoadTextFolder(
        folder=str(examples_dir),
        include_subdirectories=False,
        extensions=[".json"],
        pattern="*.json" if not single_file else single_file,
    )

    # Create the agent to generate the thumbnail prompt
    agent = Agent(
        prompt=load_text.out.text,
        model=LanguageModel(
            type="language_model",
            id="gpt-5.2",
            provider=Provider.OpenAI,
        ),
        system=THUMBNAIL_SYSTEM_PROMPT,
        tools=[],
        max_tokens=512,
    )
    thumbnail = NanoBanana(
        prompt=agent.out.text,
    )
    output_path = RegexReplace(
        text=load_text.out.path,
        pattern=r".*/(.*)\.json$",
        replacement=r"\1.png",
    )
    save_image = SaveImageFile(
        image=thumbnail.output,
        folder="src/nodetool/assets/nodetool-huggingface",
        filename=output_path.output,
        sync_mode="zip_all"
    )

    return create_graph(save_image)


def get_examples_directory() -> Path:
    """Get the path to the nodetool-huggingface examples directory."""
    script_dir = Path(__file__).parent
    examples_dir = script_dir / ".." / "src" / "nodetool" / "examples" / "nodetool-huggingface"
    return examples_dir.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Generate thumbnail image prompts for NodeTool workflows"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="thumbnail_prompts.json",
        help="Output file for results (default: thumbnail_prompts.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running the agent",
    )
    parser.add_argument(
        "--single",
        type=str,
        help="Process only a single workflow file",
    )

    args = parser.parse_args()

    examples_dir = get_examples_directory()
    print(f"Examples directory: {examples_dir}")

    if not examples_dir.exists():
        print(f"Error: Examples directory not found: {examples_dir}")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN] Would process files using LoadTextFolder node")
        print(f"  Folder: {examples_dir}")
        print(f"  Extensions: .json")
        if args.single:
            print(f"  Pattern: {args.single}")
        return

    print("\nUsing LoadTextFolder node to stream and process files...")

    try:
        # Create the workflow graph
        graph = create_workflow_graph(examples_dir, args.single)

        # Run the graph
        print("Running workflow...")
        result = run_graph(graph)

        # Extract results - Collect returns a list
        results_data = result.get("thumbnail_prompts", [])
        print(f"Received {len(results_data)} items from Collect node")

        # Parse results into structured format
        for r in results_data:
            print(r)


    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
