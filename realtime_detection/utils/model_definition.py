"""
Lightweight CNN Model Definition for Sneeze Detection

This model uses Depthwise Separable Convolutions for efficiency,
making it suitable for embedded systems like Raspberry Pi and Jetson Nano.

Architecture:
    Input: (batch, 1, 60, 63) - MFCC with deltas
    Output: (batch, 2) - [not_sneeze_prob, sneeze_prob]

Total Parameters: ~50K (0.2 MB in FP32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightSneezeCNN(nn.Module):
    """
    Lightweight CNN for Sneeze Detection

    Uses Depthwise Separable Convolutions to reduce parameters
    and computational cost while maintaining accuracy.
    """

    def __init__(self, input_height=60, input_width=63, num_classes=2):
        """
        Initialize the model

        Args:
            input_height (int): Height of input MFCC features (default: 60)
            input_width (int): Width of input time frames (default: 63)
            num_classes (int): Number of output classes (default: 2)
        """
        super(LightweightSneezeCNN, self).__init__()

        # Depthwise Separable Conv Block 1
        self.conv1_dw = nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1)
        self.conv1_pw = nn.Conv2d(1, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Depthwise Separable Conv Block 2
        self.conv2_dw = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.conv2_pw = nn.Conv2d(32, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Depthwise Separable Conv Block 3
        self.conv3_dw = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.conv3_pw = nn.Conv2d(64, 128, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, 60, 63)

        Returns:
            torch.Tensor: Output logits of shape (batch, 2)
        """
        # Block 1
        x = self.conv1_dw(x)
        x = self.conv1_pw(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2_dw(x)
        x = self.conv2_pw(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3_dw(x)
        x = self.conv3_pw(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten

        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test model
    model = LightweightSneezeCNN()

    # Test forward pass
    dummy_input = torch.randn(1, 1, 60, 63)
    output = model(dummy_input)

    print(f"Model: LightweightSneezeCNN")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Model size (FP32): {model.get_num_params() * 4 / 1024 / 1024:.2f} MB")
