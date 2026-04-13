import torch
from diffusers import Flux2KleinPipeline, PipelineQuantizationConfig
from transformers import BitsAndBytesConfig

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

cache_dir = r"D:\Users\14623\Documents\Coding\ml\flux2\flux_cache\models--black-forest-labs--FLUX.2-klein-4B\snapshots\5e67da950fce4a097bc150c22958a05716994cea"

# Wrap BitsAndBytesConfig in PipelineQuantizationConfig explicitly
quantization_config = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs=bnb_config.to_dict())

pipe = Flux2KleinPipeline.from_pretrained(
    cache_dir,
    quantization_config=quantization_config,
    torch_dtype=dtype,
    enable_profiler=enable_profiler,
)

print(pipe.transformer.config)
print("double blocks:", len(pipe.transformer.transformer_blocks))
print("single blocks:", len(pipe.transformer.single_transformer_blocks))

pipe.enable_model_cpu_offload()


from PIL import Image

prompt = "a cat on a table"
block_radius = 16
attn_heatmap_dir = f"./maps_{prompt.replace(' ', '_')}{'_block_' + str(block_radius) if block_radius else ''}"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0),
    save_attn_heatmaps=True,
    attn_heatmap_dir=attn_heatmap_dir,
    attn_block_radius=16,
).images[0]
image.save("flux-klein.png")