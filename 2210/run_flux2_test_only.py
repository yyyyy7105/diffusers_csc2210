import torch
from diffusers import Flux2KleinPipeline, PipelineQuantizationConfig
from transformers import BitsAndBytesConfig

print(torch.__version__)
print(torch.cuda.get_device_properties(0).shared_memory_per_block)

device = "cuda"
print(f"{torch.cuda.is_available()=}")
dtype = torch.float16
enable_profiler = False
print(f"{enable_profiler=}")

# Configure 4-bit quantization using BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype
)

cache_dir = "black-forest-labs/FLUX.2-klein-4B"

# Wrap BitsAndBytesConfig in PipelineQuantizationConfig explicitly
quantization_config = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs=bnb_config.to_dict())

pipe = Flux2KleinPipeline.from_pretrained(
    cache_dir,
    quantization_config=quantization_config,
    torch_dtype=dtype,
    enable_profiler=enable_profiler,
)

pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")

print(pipe.transformer.config)
print("double blocks:", len(pipe.transformer.transformer_blocks))
print("single blocks:", len(pipe.transformer.single_transformer_blocks))

# pipe.enable_model_cpu_offload()

from PIL import Image
from diffusers.models.transformers.transformer_flux2 import LinearBlockRadius, StepBlockRadius

prompt = "photo-realistic, a cat saying hello world"
save_attn_heatmaps = False
# Block radius options (pick one):
#   int                — constant radius every step
#   LinearBlockRadius  — shrink from large (diffuse) to small (focused) as steps progress
#   StepBlockRadius    — no blocking for early steps, then fixed radius
#   lambda             — fully custom schedule
#   None               — no blocking
# block_radius = LinearBlockRadius(start=32, end=4)
# block_radius = StepBlockRadius(start_step=1, radius=8)
block_radius = 2
# block_radius = None

attn_heatmap_dir = f"./maps_{prompt.replace(' ', '_')}{'__block_' + str(block_radius) if block_radius else ''}/" if save_attn_heatmaps else "./"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0),
    save_attn_heatmaps=save_attn_heatmaps,
    attn_heatmap_dir=attn_heatmap_dir,
    attn_block_radius=block_radius,
).images[0]
image.save(attn_heatmap_dir + "flux-klein.png")