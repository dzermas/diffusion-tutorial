import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from accelerate import Accelerator

from tqdm.auto import tqdm

from dataclasses import dataclass


from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


@dataclass
class TrainingConfig:
    image_size = 512  # CHANGE UNET LAYERS IF YOU CHANGE THIS
    train_batch_size = 1
    eval_batch_size = 1  # how many images to sample during evaluation
    num_epochs = 100
    num_inference_steps = 1000  # the number of steps to run the model during inference
    gradient_accumulation_steps = 16
    learning_rate = 0.00001
    lr_warmup_steps = 10
    save_progress_epochs = 2
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "corn-rows-normalized-gradacc-512-morelayersUNET"  # the model name locally and on the HF Hub
    data_dir = "/home/ubuntu/diffusion-tutorial/data"
    mean = [0.3925, 0.3505, 0.3117]
    std = [0.1126, 0.1050, 0.0876]
    seed = 0 


class SimpleImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    # model_training_path = config.output_dir
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pipeline = DDPMPipeline.from_pretrained(model_training_path).to(device)

    # pipeline.scheduler = DDIMScheduler.from_pretrained(f'{model_training_path}/scheduler')
    # images = [pipeline(num_inference_steps=40).images[0] for i in range(4)]
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
        num_inference_steps=config.num_inference_steps
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=2, cols=2)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_progress_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                pipeline.save_pretrained(config.output_dir, safe_serialization=False)
                evaluate(config, epoch, pipeline)                


if __name__ == "__main__":
    config = TrainingConfig()

    # Create transformation concatenation with random horizontal and vertical flips
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.CenterCrop(460),
        torchvision.transforms.Resize(config.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=config.mean, std=config.std)
    ])

    # Load the dataset
    dataset = SimpleImageDataset(config.data_dir, transform=transform)
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4,
    )

    # Output channels
    block_out_channels = (
        config.image_size // 4, config.image_size // 4, 
        config.image_size // 2, config.image_size // 2, 
        config.image_size, config.image_size
    )
    # block_out_channels = [config.image_size // 8, config.image_size // 4, config.image_size // 2, config.image_size]
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        # block_out_channels=block_out_channels,  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        ),
        up_block_types=(
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        dropout=0.0
    )
    model.enable_xformers_memory_efficient_attention()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)