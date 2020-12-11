# read a csv file using numpy array, split it into labeled and unlabeled sets.
# split the data set uniformly at random into K subsets
# calculate mean and covariance of each of those subsets

import matplotlib.pyplot as plt
import numpy as np
import os

UNLABELED = -1


def data_load(csv_path):
    # Load headers
    # Unsupervised, no labels provided
    with open(csv_path, 'r') as csv_file_header:
        headers = csv_file_header.readline().strip().split(',')
        print(headers)
    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    print("x_cols ", x_cols)
    print("z_cols ", z_cols)

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    print("x_shape", x.shape)
    print("z_shape", z.shape)

    # print("x ", x)
    # print("z 1->", z)
    # axis =0, each column
    # axis =1, each row

    if z.ndim == 1:
        z = np.expand_dims(z, axis=1)
        # print("z 2->", z)

    return x, z


def initialize_mu_sigma(x_train, K):
    # split example data points uniformly at random into K groups
    # calculate sample mean and covariance for each group

    n, d = x_train.shape
    group = np.random.choice(K, n)
    print('group', group.shape)
    for g in range(K):
        print('g-> ',g)
    # axis =0, mean of each column
    # axis =1, mean of each row
    mu = [np.mean(x_train[group == g, :], axis=0) for g in range(K)]
    sigma = [np.cov(x_train[group == g, :].T) for g in range(K)]
    print("mu", mu )
    print("sigma", sigma)


def main():
    # load data set
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = data_load(train_path)

    # split into labeled and unlabeled
    labeled_idxs = (z_all != UNLABELED).squeeze()
    print(labeled_idxs.shape)
    x_tilde = x_all[labeled_idxs, :] # labeled example
    z_tilde = z_all[labeled_idxs, :] # corresponding labels
    x = x_all[~labeled_idxs, :]      # unlabeled
    print("x_tilde shape", x_tilde.shape)
    print("z_tilde shape", z_tilde.shape)
    print("x_ shape", x.shape)

    # split the dataset uniformly at random into 3 subsets
    # and calculate mean and covariance matrix for each subset
    initialize_mu_sigma(x, 3)


if __name__ == '__main__':
    main()





