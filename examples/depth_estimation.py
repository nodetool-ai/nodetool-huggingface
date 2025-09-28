import asyncio
import os
from nodetool.dsl.graph import graph, run_graph
from nodetool.dsl.nodetool.constant import Image
from nodetool.dsl.nodetool.image import SaveImageFile
from nodetool.metadata.types import (
    ImageRef,
    # HFDepthEstimation can be imported if specific model selection is needed
)
from nodetool.dsl.huggingface.depth_estimation import DepthEstimation

dirname = os.path.dirname(__file__)
image_path = os.path.join(dirname, "test.jpg")
output_dir = dirname

# Create the graph structure
g = SaveImageFile(
    image=DepthEstimation(
        image=Image(value=ImageRef(uri=image_path, type="image")),
    ),
    folder=output_dir,
    filename="depth_map.png",  # Use PNG for potentially better quality depth map
)


# Run the graph
async def main():
    await run_graph(graph(g))


if __name__ == "__main__":
    print(f"Processing image: {image_path}")
    print(f"Saving depth map to: {os.path.join(output_dir, 'depth_map.png')}")
    asyncio.run(main())
    print("Depth estimation complete.")
