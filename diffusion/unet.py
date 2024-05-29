import torch
from torch import nn
from torch.functional import F


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Input(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.in_conv = DoubleConvolution(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        return self.in_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvolution(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        return self.down(x)
    

class Up(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        )
        self.conv = DoubleConvolution(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = F.pad(x1, [0, x2.size(3) - x1.size(3), 0, x2.size(2) - x1.size(2)])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConvolution(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.out_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.out_conv(x)
    


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, first_out_channels=64, num_layers=4):
        """Initialize the UNet model.
        
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            first_out_channels (int): The number of output channels of the first convolution.
            num_layers (int): The number of layers in the UNet.
        """
        super().__init__()
        self.input = Input(in_channels=in_channels, out_channels=first_out_channels)

        self.downs = nn.ModuleList()
        for i in range(num_layers):
            in_channs = first_out_channels * 2 ** i
            out_channs = in_channs * 2
            self.downs.append(Down(in_channs, out_channs))

        self.ups = nn.ModuleList()
        for i in range(num_layers):
            in_channs = out_channs
            out_channs = in_channs // 2
            self.ups.append(Up(in_channs, out_channs))

        self.out_conv = nn.Conv2d(out_channs, out_channels, 1)

    def forward(self, x):
        skips = []
        x = self.input(x)
        for down in self.downs:
            skips.append(x)
            x = down(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        x = self.out_conv(x)
        return x