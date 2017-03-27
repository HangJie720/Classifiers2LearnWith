"""Some tools for displaying information to users."""

from __future__ import division, print_function, absolute_import
import os
import random

import numpy as np
from scipy.io import loadmat


def split_data(X, Y, testpart, validpart=0, shuffle=True):

    """Split data into training, validation, and test sets.
    Args:
        X: any sliceable iterable
        Y: any sliceable iterable
        validpart: int or float proportion
        testpart: int or float proportion
        shuffle: bool
    """
    m = len(Y)

    # shuffle data
    if shuffle:
        permutation = range(m)
        random.shuffle(permutation)
        X = X[permutation]
        Y = Y[permutation]

    if 0 < validpart < 1 and 0 < testpart < 1:
        m_valid = int(validpart * m)
        m_test = int(testpart * m)
        m_train = len(Y) - m_valid - m_test
    else:
        m_valid = validpart
        m_test = testpart
        m_train = m - m_valid - m_test

    X_train = X[:m_train]
    Y_train = Y[:m_train]

    X_valid = X[m_train: m_train + m_valid]
    Y_valid = Y[m_train: m_train + m_valid]

    X_test = X[m_train + m_valid: len(X)]
    Y_test = Y[m_train + m_valid: len(Y)]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def k21hot(labels, k):
    """Labels should be a subset of {0, 1, 2, ...,  k}."""
    hot_labels = np.zeros((len(labels), k), dtype=labels.dtype)
    try:
        hot_labels[np.arange(len(labels)), labels] = 1
    except IndexError:
        hot_labels[np.arange(len(labels)), labels.astype(int)] = 1
    return hot_labels


def parentdir(path_, n=1):
    for i in range(n):
        path_ = os.path.dirname(path_)
    return path_

_default_ytype = 'float32'


def load_data(dataset, testpart=0.2, validpart=0.2, one_hot=False,
              flatten=True, shuffle=True, center=False, normalize=False,
              xtype='float32', ytype=_default_ytype,
              force_index_friendly_labels=False, resize=False):
    assert 0 < validpart + testpart < 1

    # Load dataset
    try:
        rootdir = parentdir(parentdir(os.path.abspath(__file__)))
        if not os.path.exists(rootdir):
            rootdir = parentdir(rootdir)
    except:  # in case I'm trying to paste into ipython
        rootdir = os.getcwd()
    datadir = os.path.join(rootdir, 'data')
    if dataset.lower() == 'usps':
        dataset_dict = loadmat(os.path.join(datadir, 'usps', 'USPS.mat'))
    elif dataset.lower() == 'mnist':
        dataset_dict = loadmat(os.path.join(datadir, 'mnist', 'MNIST.mat'))
    elif dataset.lower() == 'notmnist_small':
        dataset_dict = loadmat(os.path.join(datadir, 'notMNIST',
                                'notMNIST_small_no_duplicates.mat'))
    elif dataset.lower() == 'notmnist_large':
        dataset_dict = loadmat(os.path.join(datadir, 'notMNIST',
                                'notMNIST_large_no_duplicates.mat'))
    else:
        dataset_dict = loadmat(dataset)

    X = dataset_dict['X']
    Y = dataset_dict['y'].flatten()

    # force index friendly labels, if requested (and and one-hot not requested)
    if force_index_friendly_labels and not one_hot:
        label2num = dict([(l, n) for n, l in enumerate(np.unique(Y))])
        Y = np.array([label2num[y] for y in Y], dtype='int8')


    # Flatten images
    if len(X.shape) == 3:
        m, h, w = X.shape
        if flatten:
            n = w*h
            X = X.reshape((m, n))
    elif len(X.shape) == 2:
        m, n = X.shape
    else:
        raise ValueError("Data must be matrix or 3-tensor.")

    # Convert labels to one-hot format
    if one_hot:
        k = len(set(Y))
        Y = k21hot(Y % k, k)

    if xtype:
        X = X.astype(xtype)
    if ytype and (not force_index_friendly_labels and ytype != _default_ytype):
        Y = Y.astype(ytype)

    # normalize data
    if normalize:
        # X = X - mean(X)
        # X = X./mean(sum((X-mean(X)).^2))
        raise NotImplementedError


    return split_data(X, Y, testpart, validpart, False)
