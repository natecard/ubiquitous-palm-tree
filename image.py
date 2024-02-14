# Investigating the use of the stable-diffusion-xl-base-1.0 model for image generation
# Also going to look at the SDXL Turbo model to compare, paying attention to memory and time usage
from datetime import datetime
import os
from diffusers import DiffusionPipeline, AutoencoderKL
from DeepCache import DeepCacheSDHelper
import torch


vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
# Directory to save the images
test_dir = os.path.join("test_images")
# Create the directory if it doesn't exist
os.makedirs(test_dir, exist_ok=True)

pipeline = DiffusionPipeline.from_pretrained(
    # Uncomment to use the SDXL Turbo model
    # "stabilityai/sdxl-turbo",
    # Comment out to use the SDXL Turbo model
    "segmind/SSD-1B",
    torch_dtype=torch.float16,
    variant="fp16",
    vae=vae,
    use_safetensors=True,
).to("mps")
helper = DeepCacheSDHelper(pipe=pipeline)
helper.set_params(
    cache_interval=5,
    cache_branch_id=0,
)
helper.enable()
# Torch._dynamo issues with MPS? (not sure)
# pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
# Does not work with custom VAE
# pipeline.enable_attention_slicing("max")


# Refiner model for XL pipeline (better quality & optional, but slower)
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("mps")


# Used to reduce GPU memory usage when using CUDA
# pipeline.enable_sequential_cpu_offload()
# Used to reduce memory overhead
# pipeline.enable_attention_slicing("max")

prompt = (
    input("Enter a prompt: ") or "A painting of an elephant in the style of Picasso."
)
neg_prompt = (
    input("Enter a negative prompt: ")
    or "follow the prompted image as closely as possible."
)
# Number of inference steps
n_steps = 50
# For SDXL to denoise the image
# high_noise_frac = 0.7

image = pipeline(
    prompt=prompt,
    negative_prompt=neg_prompt,
    # If swapped to SDXL Turbo, use the following lines
    # num_inference_steps=n_steps,
    # output_type="image",
    # change next line to images[0] if not using the refiner model
).images

# Uncomment to use the refiner model for better quality
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    image=image,
).images[0]

# Hugging Face utils API not working currently
# Make a grid out of the images
# images = image[0:]
# image_grid = make_image_grid(images, rows=4, cols=4)

# Save the images
now = datetime.now()
format_time = now.strftime("%m_%d_%h_%m_%s")
# Save images to the directory with the current date and time
image.save(f"{test_dir}/{now}_image.png")
