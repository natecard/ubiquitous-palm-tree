# Investigating the use of the stable-diffusion-xl-base-1.0 model for image generation
# Also going to look at the SDXL Turbo model to compare, paying attention to memory and time usage

from diffusers import DiffusionPipeline, AutoencoderKL
import torch


vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)


pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("mps")

# Refiner model for XL pipeline (better quality & optional, but slower)
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("mps")


# pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
# helper = DeepCacheSDHelper(pipe=pipeline)
# helper.set_params(
#     cache_interval=3,
#     cache_branch_id=0,
# )
# helper.enable()
# Used to reduce GPU memory usage
# pipeline.enable_sequential_cpu_offload()
# Used to reduce memory overhead
# pipeline.enable_attention_slicing("max")

prompt = (
    input("Enter a prompt: ") or "A painting of an elephant in the style of Picasso."
)

n_steps = 40
high_noise_frac = 0.7

image = pipeline(
    prompt=prompt,
    num_inference_steps=n_steps,
    output_type="latent",
).images

image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    image=image,
).images[0]

image

image.save(f"{prompt[0:6]}.png")
