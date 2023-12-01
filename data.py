from torch.utils.data import DataLoader
import os
import os.path
import numpy as np
import sys
import csv
from sklearn.preprocessing import MinMaxScaler

import pickle
import torch
import torch.utils.data as data
from scipy.optimize import leastsq

from itertools import permutations

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# Define a function to solve the sparse representation
def sparse_representation(csi, dictionary):
    # Use least squares to solve the sparse representation
    x0 = np.zeros(dictionary.shape[1])
    result = leastsq(lambda x: csi - np.dot(dictionary, x), x0)
    return result[0]

# Define a function to update the filter coefficients
def update_filter_coefficients(filter_coefficients, gradient, step_size):
    # Use gradient descent to update the filter coefficients
    filter_coefficients -= step_size * gradient
    return filter_coefficients

def custom_minmax_scaler(data, us):
    scaler = MinMaxScaler(feature_range=(0, 1))
    if us:
        data = scaler.fit_transform(data)
        return data.reshape(-1, 1, 17, 30)
    else:
        return data

def get_dataloader(args):
    x = []
    dataset_path = args.root,
    pr = args.pr,
    sa = args.sa,
    train_trainData = os.path.join(dataset_path[0], 'train/csi_cdy_am_train_6ac.csv')
    filedata = open(train_trainData)
    readerdata = csv.reader(filedata)
    
    # Initialize the filter coefficients
    filter_coefficients = np.random.rand(17, 30)

    for ind, k in enumerate(readerdata):
        k = list(map(float, k))
        k = np.array(k)
        k = k.reshape(1, 1, 17, 30)
        x.append(k)
    trainData = np.array(x).reshape(-1, 1, 1, 17, 30)

    "load testdata"
    m = []
    pathDir = os.path.join(dataset_path[0], 'test/csi_cdy_am_test_6ac.csv')
    filedata = open(pathDir)
    readerdata = csv.reader(filedata)
    for ind, k in enumerate(readerdata):
        k = list(map(float, k))
        k = np.array(k)
        k = k.reshape(1, 1, 17, 30)
        m.append(k)
    valData = np.array(m).reshape(-1, 1, 1, 17, 30)

    data_pro = np.concatenate([trainData, valData], axis=0)

    # normalization
    batch_size = data_pro.shape[0]
    normalized_data = np.zeros_like(data_pro)

    for i in range(batch_size):
        batch_data = data_pro[i].reshape(-1, 1)
        normalized_data[i] = custom_minmax_scaler(batch_data, pr)

    # SRAF
    rs_csis = []
    if sa:
        for i in range(normalized_data.shape[0]):
            csi = normalized_data[0, 0, :, :]
            # Construct the dictionary
            dictionary = np.array([np.roll(csi.flatten(), j) for j in range(csi.size)]).T
            # Solve the sparse representation
            sparse_rep = sparse_representation(csi.flatten(), dictionary)
            # Calculate the gradient
            gradient = 2 * np.dot(dictionary.T, np.dot(dictionary, filter_coefficients.flatten()) - csi.flatten())
            # Update the filter coefficients
            filter_coefficients = update_filter_coefficients(filter_coefficients.flatten(), gradient, step_size=0.01)
            # Reconstruct the CSI signal
            rs_csi = np.dot(dictionary, sparse_rep).reshape(csi.shape)
            rs_csis.append(rs_csi)
        rs_csis = np.array(rs_csis).reshape(batch_size, 1, 1, 17, 30)
    else:
        rs_csis = np.array(normalized_data).reshape(batch_size, 1, 1, 17, 30)

    trainData = rs_csis[0:trainData.shape[0]]
    valData = rs_csis[trainData.shape[0]:]

    trainData1 = np.copy(trainData)
    train_trainData = np.concatenate([trainData, trainData1], axis=1) 
    valData1 = np.copy(valData)
    val_trainData = np.concatenate([valData, valData1], axis=1) 

    train_trainData = torch.tensor(train_trainData)
    print("the train_dataset shape is:", train_trainData.size())
    val_trainData = torch.tensor(val_trainData)
    print("the shape of val_trainData:", val_trainData.size())

    l = []
    pathDir = os.path.join(dataset_path[0], 'train/kinect_xy_train_cdy_6ac.csv')
    filetarget = open(pathDir)
    readerdata = csv.reader(filetarget)
    for ind, k in enumerate(readerdata):
        k = list(map(float, k))
        l.append(k)
    train_trainTarget = np.array(l)

    n = []
    pathDir = os.path.join(dataset_path[0], 'test/kinect_xy_test_cdy_6ac.csv')
    filetarget = open(pathDir)
    readerdata = csv.reader(filetarget)
    for ind, k in enumerate(readerdata):
        k = list(map(float, k))
        n.append(k)
    val_trainTarget = np.array(n)

    train_trainTarget = torch.tensor(train_trainTarget)
    train_trainTarget = train_trainTarget.float()
    print("the train_dataset_target shape is:", type(train_trainTarget), train_trainTarget.size())
    train_dataset = torch.utils.data.TensorDataset(train_trainData, train_trainTarget) 
    train_loader = torch.utils.data.DataLoader(   
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False
    )

    val_trainTarget = torch.tensor(val_trainTarget)
    val_trainTarget = val_trainTarget.float()
    print("the val_dataset_target shape is:", type(val_trainTarget), val_trainTarget.size())
    val_dataset = torch.utils.data.TensorDataset(val_trainData, val_trainTarget)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )


    return train_loader, val_loader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--model_names', type=str, nargs='+', default=['conti', 'conti'])

    args = parser.parse_args()
    train_loader, val_loader = get_dataloader(args)
    
    for img, label in train_loader:
        break
    for img, label in val_loader:
        print(img.shape, label.shape)

        break