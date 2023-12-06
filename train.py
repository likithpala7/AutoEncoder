import torch.nn as nn
import torch.nn.functional as F
import torch
from DataLoader import DataBatch, concatenate_data
from Transformer import Encoder
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


data_dir = 'cifar-10-batches-py'

data, labels = concatenate_data(data_dir)
data = np.transpose(data, (0, 3, 1, 2))

with open(os.path.join(data_dir, 'batches.meta'), 'rb') as f:
    meta_data = pickle.load(f, encoding='bytes')
    label_names = meta_data[b'label_names']

model = Encoder()
model.to(torch.double)
# for x, y in DataBatch(data, labels, 64):\
# print(data[0])
print(model.forward(torch.from_numpy(data[10]*1.0)))