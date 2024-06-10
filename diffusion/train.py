import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from diffusion_model import DiffusionModel, add_noise, remove_noise
from unet import UNet


def train(model, data_loader, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    model.train()
    device = next(model.parameters()).device
    for epoch in range(epochs):
        loss = torch.tensor(0.0)
        counter = 1
        for images, _ in data_loader:
            images = torch.stack([image.to(device) for image in images])
            loss = 0
            noisy_images = images
            for t in reversed(range(100)):
                noisy_images = add_noise(noisy_images, t/100.0)
                if t == 0:
                    continue
                reconstructed = remove_noise(noisy_images, t/100.0, model)
                # plot the noisy and reconstructed images side by side with the t value
                # Create a plot with 1 row and 3 columns the 3rd column is for the difference between the images
                # plt.subplot(1, 3, 1)
                # plt.title(f"Noisy Image, t={t}")
                # plt.imshow(noisy_images[0].cpu().detach().numpy().reshape(28, 28))
                # plt.subplot(1, 3, 2)
                # plt.title(f"Reconstructed Image, t={t}")
                # plt.imshow(reconstructed[0].cpu().detach().numpy().reshape(28, 28))
                # plt.subplot(1, 3, 3)
                # plt.title(f"Difference Image, t={t}")
                # plt.imshow(noisy_images[0].cpu().detach().numpy().reshape(28, 28) - reconstructed[0].cpu().detach().numpy().reshape(28, 28))
                # plt.show()
                loss += criterion(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss {loss.item()}, {counter}/{len(data_loader)}")
            counter += 1

train_data = MNIST(root='./', train=True, transform=ToTensor(), download=True)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionModel(in_channels=1, out_channels=1).to(device)
train(model, train_loader)
torch.save(model.state_dict(), "diffusion_model.pth")

