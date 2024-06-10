
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

# Create transformation concatenation with random horizontal and vertical flips
transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(460),
    torchvision.transforms.Resize(512),
    torchvision.transforms.ToTensor(),
])

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
        return {"image": image}


# Load the dataset
data_dir = "/home/ubuntu/diffusion-tutorial/data"
dataset = SimpleImageDataset(data_dir, transform=transform)

# Custom collate function for DataLoader
def collate_fn(batch):
    images = [item['image'] for item in batch]
    images = torch.stack([torch.tensor(image).permute(2, 1, 0) for image in images])  # Convert to (C, H, W) format
    return images

train_dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)


# Calculate mean and std for the dataset
mean = 0.
std = 0.
nb_samples = 0.

for step, data in enumerate(train_dataloader):
    mean += data.mean([0, 1, 2])
    std += data.std([0, 1, 2])
    nb_samples += data.size(0)

mean /= nb_samples
std /= nb_samples

print(f'Mean: {mean}')
print(f'Std: {std}')