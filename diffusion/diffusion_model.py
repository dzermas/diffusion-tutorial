
import torch
from torch import nn

import matplotlib.pyplot as plt

from unet import UNet

class DiffusionModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = UNet(in_channels=in_channels, out_channels=out_channels, first_out_channels=32, num_layers=2)

    def forward(self, x):
        return self.model(x)
    
def noise_scheduler(t, beta_start=0.01, beta_end=0.04):
    return torch.tensor(beta_start + (beta_end - beta_start) * t)

def add_noise(x, t, beta_start=0.01, beta_end=0.04):
    beta = noise_scheduler(t, beta_start, beta_end)
    # Generates noise with the same shape as the batch of images x
    noise = torch.randn_like(x) * torch.sqrt(beta)
    return x * torch.sqrt(1 - beta) + noise

def remove_noise(x, t, model, beta_start=0.01, beta_end=0.04):
    beta = noise_scheduler(t, beta_start, beta_end)
    # Model predicts the noise directly from the noisy images
    pred_noise = model(x)
    # Recovers the batch of images by denoising
    return (x - torch.sqrt(beta) * pred_noise) / torch.sqrt(1 - beta)
