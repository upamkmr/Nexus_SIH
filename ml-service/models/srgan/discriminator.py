"""
Sentinel-2 Super-Resolution Discriminator Network
Validates multi-spectral spatial textures and channel relationships.
"""

try:
    import torch
    import torch.nn as nn

    class DiscriminatorBlock(nn.Module):
        def __init__(self, in_features, out_features, stride=1, normalize=True):
            super().__init__()
            layers = [nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)

    class SRGANDiscriminator(nn.Module):
        """
        VGG-style discriminator for evaluating 4-band / 3-band HR patches.
        """
        def __init__(self, in_channels=4):
            super().__init__()
            self.net = nn.Sequential(
                DiscriminatorBlock(in_channels, 64, stride=1, normalize=False),
                DiscriminatorBlock(64, 64, stride=2),
                DiscriminatorBlock(64, 128, stride=1),
                DiscriminatorBlock(128, 128, stride=2),
                DiscriminatorBlock(128, 256, stride=1),
                DiscriminatorBlock(256, 256, stride=2),
                DiscriminatorBlock(256, 512, stride=1),
                DiscriminatorBlock(512, 512, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, 1)
            )

        def forward(self, x):
            return self.net(x)

except ImportError:
    nn = None
    torch = None
    DiscriminatorBlock = None
    SRGANDiscriminator = None
