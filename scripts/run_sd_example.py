#!/usr/bin/env python3
"""
Example script to run Stable Diffusion XL image generation.
This script demonstrates basic usage of the nodetool-huggingface package.
"""
import asyncio
import shutil
import sys
from pathlib import Path

# Add parent directory to path for imports when running from source
# This allows the script to work both when installed via pip and when run directly
# from the repository. The startup script installs the package with 'pip install -e .'
# so this path modification is typically only needed during local development.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodetool.metadata.types import HFStableDiffusionXL
from nodetool.dsl.huggingface.text_to_image import StableDiffusionXL
from nodetool.workflows.processing_context import ProcessingContext


async def generate_image():
    """Generate an image using Stable Diffusion XL."""
    print("=" * 60)
    print("Stable Diffusion XL Example")
    print("=" * 60)

    # Create the processing context
    context = ProcessingContext(user_id="test_user", auth_token="")

    # Create and configure the StableDiffusionXL node
    print("\nConfiguring Stable Diffusion XL node...")
    sdxl = StableDiffusionXL()

    # Use a quantized model that's faster and requires less VRAM
    sdxl.model = HFStableDiffusionXL(
        type="hf.stable_diffusion_xl",
        repo_id="nunchaku-tech/nunchaku-sdxl",
        path="svdq-int4_r32-sdxl.safetensors",
    )

    # Configure generation parameters
    sdxl.prompt = (
        "A serene mountain landscape at sunset, highly detailed, photorealistic"
    )
    sdxl.negative_prompt = "blurry, low quality, distorted, ugly"
    sdxl.height = 768
    sdxl.width = 768
    sdxl.num_inference_steps = 20
    sdxl.guidance_scale = 7.5
    sdxl.seed = 42
    sdxl.enable_cpu_offload = True  # Enable CPU offload for better memory management

    print(f"\nPrompt: {sdxl.prompt}")
    print(f"Size: {sdxl.width}x{sdxl.height}")
    print(f"Steps: {sdxl.num_inference_steps}")
    print(f"Model: {sdxl.model.repo_id}/{sdxl.model.path}")

    # Generate the image
    print("\nGenerating image...")
    print("(This may take a few minutes on first run as the model downloads)")

    result = await sdxl.process(context)

    # Save the generated image
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sd_example_output.png"

    print("\nImage generated successfully!")
    print(f"Saving to: {output_path}")

    # The result contains an ImageRef, we need to save it
    if hasattr(result, "uri"):
        # If it's an ImageRef with a URI, copy the file
        if result.uri.startswith("file://"):
            src_path = result.uri[7:]  # Remove 'file://' prefix
            shutil.copy(src_path, output_path)
            print(f"Output saved to {output_path}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

    return output_path


def main():
    """Main entry point."""
    try:
        asyncio.run(generate_image())
        return 0
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
