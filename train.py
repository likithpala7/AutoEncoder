import torch.nn as nn
import torch.nn.functional as F
import torch
from DataLoader import DataBatch, concatenate_data
from AutoEncoder import AutoEncoder
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
data = data/255.0

with open(os.path.join(data_dir, 'batches.meta'), 'rb') as f:
    meta_data = pickle.load(f, encoding='bytes')
    label_names = meta_data[b'label_names']


ae = AutoEncoder()
ae = ae.double()
ae.to(device)

optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-6)
loss_fn = nn.MSELoss()

batchsize = 128
epochs = 10
losses = []
for epoch in range(epochs):
    for train_x, _ in tqdm(DataBatch(data, labels, batchsize), total=len(data)//batchsize):
        train_x = train_x.to(device)
        optimizer.zero_grad()
        output = ae.forward(train_x)
        loss = loss_fn(output, train_x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

plt.plot(losses)
plt.show()

torch.save(ae.state_dict(), 'autoencoder.pth')