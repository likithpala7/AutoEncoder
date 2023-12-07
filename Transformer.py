import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, (3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 32, (3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(32, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2, 2)),
            nn.ConvTranspose2d(32, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2, 2)),
            nn.ConvTranspose2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2, 2)),
            nn.ConvTranspose2d(64, 3, (3, 3), padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            Encoder(),
            Decoder()
        )

    def forward(self, x):
        x = self.model(x)
        return x