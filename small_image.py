# Description: This script is used to generate images using the Wuerstchen model.
from diffusers import (
    WuerstchenDecoderPipeline,
    WuerstchenPriorPipeline,
)
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
from datetime import datetime
import os
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.version.hip:
    device = torch.device("hip")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
    device = torch.device("cpu")

# Directory to save the images
test_dir = os.path.join("test_images")
# Create the directory if it doesn't exist
os.makedirs(test_dir, exist_ok=True)

prior_pipeline = WuerstchenPriorPipeline.from_pretrained(
    "warp-ai/wuerstchen-prior", torch_dtype=torch.float16
).to(device)
decoder_pipeline = WuerstchenDecoderPipeline.from_pretrained(
    "warp-ai/wuerstchen", torch_dtype=torch.float16
).to(device)

prompt = (
    input("Enter a prompt: ") or "A painting of an elephant in the style of Picasso."
)
neg_prompt = (
    # input("Enter a negative prompt: ") or
    "photorealism, low resolution, bad quality, low quality, bad resolution, bad lighting, bad composition, bad framing, bad focus, bad exposure, bad color, bad contrast, bad saturation, bad hue, bad brightness, bad sharpness, bad noise, bad artifacts, bad distortion, bad blur, bad grain, bad vignetting, bad moire, bad chromatic aberration, bad fringing, bad purple fringing, bad green fringing, bad red fringing, bad blue fringing, bad yellow fringing, bad cyan fringing, bad magenta fringing, bad white balance, bad color balance, bad color cast, bad color temperature, bad color grading, bad color correction, bad color enhancement, bad color manipulation, bad color filtering, bad color processing, bad color mapping, bad color space, bad color model, bad color profile, bad color gamut, bad color depth, bad color bit depth, bad color resolution, bad color accuracy, bad color fidelity, bad color reproduction, bad color management, bad color science, bad color theory, bad color psychology, bad color symbolism, bad color harmony, bad color contrast, bad color scheme, bad color wheel, bad color theory, bad color mixing, bad color blending, bad color matching, bad color combination, bad color composition, bad color theory, bad color psychology, bad color symbolism, bad color harmony, bad color contrast, bad color scheme, bad color wheel, bad color theory, bad color mixing, bad color blending, bad color matching, bad color combination, bad color composition, bad color theory, bad color psychology, bad color symbolism, bad color harmony, bad color contrast, bad color scheme, bad color wheel, bad color theory, bad color mixing, bad color blending, bad color matching, bad color combination, bad color composition, bad color theory, bad color psychology, bad color symbolism, bad color harmony, bad color contrast, bad color scheme, bad color wheel, bad color theory, bad color mixing, bad color blending, bad color matching, bad color combination, bad color composition, bad color theory, bad color psychology, bad color symbolism, bad color harmony, bad color contrast, bad color scheme, bad color wheel, bad color theory, bad color mixing, bad color blending, bad color matching, bad color combination, bad color composition, bad color theory, bad color psychology, bad color symbolism, bad color harmony, bad color contrast, bad color scheme, bad color wheel, bad color theory, bad color mixing, bad color blending"
)
# Number of inference steps
n_steps = 50
num_images = 1
num_inference_steps = n_steps

prior_output = prior_pipeline(
    prompt=prompt,
    height=1024,
    width=1536,
    timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    negative_prompt=neg_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=num_images,
)

decoder_output = decoder_pipeline(
    image_embeddings=prior_output.image_embeddings,
    prompt=prompt,
    negative_prompt=neg_prompt,
    prior_guidance_scale=4.0,
    guidance_scale=0.0,
    output_type="pil",
).images[0]

# Save the images
now = datetime.now()
format_time = now.strftime("%m_%d_%h_%m_%s")
# Save images to the directory with the current date and time
decoder_output.save(f"{test_dir}/{now}_image.png")
