import torch.nn as nn
import torch.nn.functional as F
import torch
from DataLoader import DataBatch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")