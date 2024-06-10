import torch
from diffusers import DDPMPipeline, DDIMScheduler

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = DDPMPipeline.from_pretrained('/home/ubuntu/diffusion-tutorial/corn-rows-normalized-correct-stats-128').to('cpu')

pipeline.scheduler = DDIMScheduler.from_pretrained('/home/ubuntu/diffusion-tutorial/corn-rows-normalized-correct-stats-128/scheduler')
image = pipeline(num_inference_steps=40).images[0]
image.save("ddpm_generated_image.png")