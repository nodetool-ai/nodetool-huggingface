import asyncio
import os
import tempfile
from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.constant import Image
from nodetool.metadata.types import (
    ImageRef,
    HFRealESRGAN,
    HFStableDiffusion,
)
from nodetool.dsl.huggingface.image_to_image import (
    RealESRGANNode,
    StableDiffusionImg2ImgNode,
)
from nodetool.dsl.nodetool.image import SaveImageFile
from nodetool.nodes.huggingface.stable_diffusion_base import StableDiffusionBaseNode

dirname = os.path.dirname(__file__)
image_path = os.path.join(dirname, "test.jpg")
temp_dir = tempfile.mkdtemp()

g = SaveImageFile(
    image=RealESRGANNode(  # Second upscaling
        image=StableDiffusionImg2ImgNode(
            init_image=RealESRGANNode(  # First upscaling
                image=Image(value=ImageRef(uri=image_path, type="image")),
                model=HFRealESRGAN(
                    repo_id="ai-forever/Real-ESRGAN",
                    path="RealESRGAN_x4.pth",
                ),
            ),
            prompt="portrait of a woman",
            negative_prompt="blur",
            model=HFStableDiffusion(
                repo_id="SG161222/Realistic_Vision_V5.1_noVAE",
                path="Realistic_Vision_V5.1_fp16-no-ema.safetensors",
            ),
            scheduler=StableDiffusionBaseNode.StableDiffusionScheduler.DPMSolverMultistepScheduler,
            num_inference_steps=100,
            strength=0.1,
            guidance_scale=3.0,
        ),
        model=HFRealESRGAN(  # Second upscaling
            repo_id="ai-forever/Real-ESRGAN",
            path="RealESRGAN_x2.pth",
        ),
    ),
    folder=dirname,
    filename="upscaled_image_4x.jpg",
)

asyncio.run(run_graph(create_graph(g)))
