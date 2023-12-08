import torch
import os
from DataLoader import concatenate_data
import numpy as np
import pickle
import matplotlib.pyplot as plt
from Transformer import AutoEncoder

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

model = AutoEncoder()
model = model.double()
model.load_state_dict(torch.load('autoencoder.pth'))

random_choice = np.random.choice(len(data))
example = np.array([data[random_choice]])
image = np.transpose(data[random_choice], (1, 2, 0))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].set_title(label_names[labels[random_choice]])

output = model(torch.from_numpy(example))[0].detach().numpy()
output = np.transpose(output, (1, 2, 0))
ax[1].imshow(output)
ax[1].set_title("Model output")
plt.show()
