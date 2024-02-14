# Investigating the use of the stable-diffusion-xl-base-1.0 model for image generation
# Also going to look at the SDXL Turbo model to compare, paying attention to memory and time usage
import os
from diffusers.utils import make_image_grid
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
# refiner = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0",
#     vae=vae,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16",
# ).to("mps")


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
# save_name = input("Save as: ") or "image.png"

n_steps = 40
high_noise_frac = 0.7

images = pipeline(
    prompt=prompt,
    num_inference_steps=n_steps,
    output_type="latent",
).images

# Make a grid out of the images
image_grid = make_image_grid(images, rows=4, cols=4)

# Save the images
test_dir = os.path("samples")
os.makedirs(test_dir, exist_ok=True)
image_grid.save(f"{test_dir}/image.png")
# if save_name.lower().endswith(".png"):
#     save_name = save_name
#     image.save(save_name)
# else:
#     save_name = save_name.lower().strip() + ".png"
#     image.save(save_name)
# image = refiner(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     image=image,
# ).images[0]
# image
