import torch
from diffusers import Flux2KleinPipeline, PipelineQuantizationConfig
from transformers import BitsAndBytesConfig

import os
from dotenv import load_dotenv
load_dotenv()


device = "cuda"
dtype = torch.float16

# Configure 4-bit quantization using BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype
)

# Wrap BitsAndBytesConfig in PipelineQuantizationConfig explicitly
quantization_config = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs=bnb_config.to_dict())

local_model_path = os.path.join(os.getcwd(), "flux_cache")
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", 
                                          quantization_config=quantization_config, 
                                          torch_dtype=dtype, 
                                          cache_dir=local_model_path)
pipe.enable_model_cpu_offload()


from PIL import Image

# initial_image = Image.open("19878f4427786ca7f886d973cfc2a4c5.png")
prompt = "Refine the image,  keep facial expression, better resolution, photo-realistic"
image = pipe(
    prompt=prompt,
    # image=initial_image,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4, # Since you are using 4 steps (likely Flux Schnell), keep this.
    generator=torch.Generator(device=device).manual_seed(0)
).images[0]

image.save("flux-klein.png")