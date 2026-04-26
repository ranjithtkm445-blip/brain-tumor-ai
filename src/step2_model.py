# src/step2_model.py
# Purpose: Define a lightweight 2D U-Net model for brain tumor segmentation.
#          Takes a (batch, 4, 240, 240) MRI image as input (4 modalities)
#          and outputs a (batch, 3, 240, 240) segmentation mask (3 tumor
#          regions: necrosis, edema, enhancing tumor).
#          Uses encoder-decoder architecture with skip connections.
#          Designed to run on CPU without requiring a GPU.
#          Prints model summary and parameter count to confirm architecture.

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(64, 128)

        # Decoder
        self.up3  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.up2  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1  = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)

        # Segmentation output head
        self.seg_output = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Segmentation mask output
        mask = self.sigmoid(self.seg_output(d1))
        return mask


if __name__ == "__main__":
    model  = UNet(in_channels=4, out_channels=3)
    total  = sum(p.numel() for p in model.parameters())
    dummy  = torch.zeros(1, 4, 240, 240)
    output = model(dummy)

    print(f"Model        : 2D U-Net")
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Total params : {total:,}")