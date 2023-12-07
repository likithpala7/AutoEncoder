import torch.nn as nn
import torch.nn.functional as F
import torch
from DataLoader import DataBatch, concatenate_data
from Transformer import Encoder, Decoder
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tqdm

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
data = np.transpose(data, (0, 3, 1, 2)).astype(np.float64)

with open(os.path.join(data_dir, 'batches.meta'), 'rb') as f:
    meta_data = pickle.load(f, encoding='bytes')
    label_names = meta_data[b'label_names']

encoder = Encoder()
encoder = encoder.double()

decoder = Decoder()
decoder = decoder.double()

out = encoder.forward(torch.from_numpy(data[0]))
out = decoder.forward(out)
print(out.shape)
# optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-6)
# loss_fn = nn.MSELoss()

# batchsize = 64

# for train_x, train_y in tqdm(DataBatch(data, labels, batchsize), total=len(data)//batchsize):
#     optimizer.zero_grad()
#     output = model.forward(train_x)
#     loss = loss_fn(output, )
