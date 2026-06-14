import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


class CifarResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10.

    Original ResNet expects larger images, so for 32x32 CIFAR-10 images:
    - conv1 is changed from 7x7 stride 2 to 3x3 stride 1
    - maxpool is removed
    - final classifier is changed to 10 classes
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.model = models.resnet18(weights=None)

        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
