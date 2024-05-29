import torch

from diffusion_model import DiffusionModel, add_noise, remove_noise


# Load the model and generate a few samples
def generate_samples(model, num_samples=5):
    model.eval()
    device = next(model.parameters()).device
    samples = torch.randn(num_samples, 1, 28, 28).to(device)
    for t in range(10):
        samples = add_noise(samples, t/1000.0)
        samples = remove_noise(samples, t/1000.0, model)
    return samples


model = DiffusionModel(in_channels=1, out_channels=1)
state_dict = torch.load("diffusion_model.pth")
model.load_state_dict(state_dict)

num_samples=10
generated_images = generate_samples(model, num_samples=num_samples)

# You can visualize these using matplotlib
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, num_samples, figsize=(10, 1))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_images[i].squeeze(0).detach().numpy(), cmap='gray')
    ax.axis('off')
plt.show()