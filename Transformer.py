import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding_mode="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding="same"),
            nn.Conv2d(64, 64, (4, 4), padding_mode="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding="same"),
            nn.Conv2d(64, 32, (3, 3), padding_mode="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding="same"),
            nn.Flatten(),
            nn.Linear(32*32*32, 512)
        )

    def forward(self, x):
        x = self.model(x)
        return x
