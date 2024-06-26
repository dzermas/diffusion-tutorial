import os
import torch
from diffusers import DDPMPipeline, DDIMScheduler

model_training_path = "corn-rows-normalized-gradacc-512"
os.makedirs(f"{model_training_path}/ddpm_generated_images", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = DDPMPipeline.from_pretrained(model_training_path).to(device)

pipeline.scheduler = DDIMScheduler.from_pretrained(f'{model_training_path}/scheduler')
for i in range(10):
    image = pipeline(num_inference_steps=50).images[0]
    image.save(f"{model_training_path}/ddpm_generated_images/{i}.png")