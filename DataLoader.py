import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
import torch


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def concatenate_data(path):
    data = []
    labels = []

    for file in os.listdir(path):
        if "data_batch" in file:
            batch_dict = unpickle(os.path.join(path, file))
            data.append(batch_dict[b'data'])
            labels.append(batch_dict[b'labels'])

    data = np.array(data).reshape(50000, 3072)
    labels = np.array(labels)

    reshaped_data = np.zeros((data.shape[0], 1024, 3))
    for i in range(len(data)):
        reshaped_data[i, :, 0] = data[i][:1024]
        reshaped_data[i, :, 1] = data[i][1024:2048]
        reshaped_data[i, :, 2] = data[i][2048:]

    reshaped_data = reshaped_data.reshape(50000, 32, 32, 3).astype(int)
    return reshaped_data, labels.flatten()


# taken from Ben Ochoa CSE 252 assignment
def DataBatch(data, label, batchsize, shuffle=True):
    # provides a generator for batches of data that yields
    # data (batchsize, 3, 32, 32) and labels (batchsize)
    # if shuffle, it will load batches in a random order
    n = data.shape[0]
    if shuffle:
        index = np.random.permutation(n)
    else:
        index = np.arange(n)
    for i in range(int(np.ceil(n/batchsize))):
        inds = index[i*batchsize: min(n, (i+1)*batchsize)]
        yield torch.from_numpy(data[inds]), torch.from_numpy(label[inds])
