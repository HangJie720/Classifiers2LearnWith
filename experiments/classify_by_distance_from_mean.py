"""
Classify by distance from mean
"""

# Standard Library Dependencies
from __future__ import division, print_function, absolute_import
from warnings import warn
from time import time
import os
from operator import itemgetter

# External Dependencies
import numpy as np
from scipy.io import loadmat
from numpy.linalg import norm as norm2
# %matplotlib inline  # use when in JUPYTER NOTEBOOK (or risk hang)
# plt.ion()  # allow ipython %run to terminate without closing figure


class distance_from_mean_classifier(object):
    def train(self, X_train, y_train):
        distinct_labels = np.array(list(distinct_labels))
        label_means = []
        sds = []
        label_variances = []
        for l in distinct_labels:
            Xl = [x for x, y in zip(X_train, y_train) if y == l]  # examples labeled l
            mean_l = np.mean(Xl, axis=0)
            label_means.append(mean_l)

            # Compute S.D. for each examples of each label
            s_l = np.std([norm2(x - mean_l) for x in Xl])
            label_variances.append(s_l)

        # Check how far away means are from each other.
        for i, mi in enumerate(label_means):
            d2i = [(j, norm2(mi - mj)) for j, mj in enumerate(label_means) if i!=j]
            j, dij = min(d2i, key=itemgetter(1))
            si, sj = np.sqrt(label_variances[i]), np.sqrt(label_variances[j])
            print("label {} :: nearest label = {} :: dist = {}) :: "
                  "sd_{} = {} :: sd_{} = {}"
                  "".format(i, j, dij, i, si, j, sj))
        self.means = np.array(label_means)
        self.sds = np.sqrt(np.array(label_variances))
        self.ordered_labels = distinct_labels

    def predict(X):
        pred_indices = np.argmin([norm2(X - m, axis=1) for m in self.means], axis=0)
        return ordered_labels[pred_indices]


########################################################################       
# User Parameters
DATASET = 'USPS'
VALIDATION_PERCENTAGE = 0
TESTING_PERCENTAGE = .3
NORMALIZE = False

assert 0 < VALIDATION_PERCENTAGE + TESTING_PERCENTAGE < 1

# Load dataset
try: 
    rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except:  # in case I'm trying to paste into ipython
    rootdir = os.getcwd()
datadir = os.path.join(rootdir, 'data')
if DATASET.lower() == 'usps':
    dataset_dict = loadmat(os.path.join(datadir, 'usps', 'USPS.mat'))
elif DATASET.lower() == 'mnist':
    dataset_dict = loadmat(os.path.join(datadir, 'mnist', 'MNIST.mat'))
elif DATASET.lower() == 'notmnist_small':
    dataset_dict = loadmat(os.path.join(datadir, 'notMNIST', 
                            'notMNIST_small_no_duplicates.mat'))
elif DATASET.lower() == 'notmnist_large':
    dataset_dict = loadmat(os.path.join(datadir, 'notMNIST', 
                            'notMNIST_large_no_duplicates.mat'))
else:
    dataset_dict = loadmat(DATASET)

X = dataset_dict['X'].astype('float32')
y = dataset_dict['y'].ravel()
distinct_labels = set(y)

# Flatten images
m, h, w = X.shape
n = w*h
X = X.reshape((m, n))

# shuffle data
from random import shuffle
permutation = range(m)
shuffle(permutation)
X = X[permutation]
y = y[permutation]

# Convert labels to one-hot format
# def onehot(data, ordered_label_set):
#     return [[(1 if l == y else 0) for l in ordered_label_set] for y in data]
# y = np.array(onehot(y, distinct_labels)).astype('float32')
# print(y.shape)

# normalize data
if NORMALIZE:
    # X = X - mean(X)
    # X = X./mean(sum((X-mean(X)).^2))
    raise NotImplementedError

m_train = int(0.6*m)
X_train = X[:m_train]
y_train = y[:m_train]

m_valid = int(0.2*m)
X_valid = X[m_train : m_train + m_valid + 1]
y_valid = y[m_train : m_train + m_valid + 1]

m_test = m - m_valid - m_train
X_test = X[m_train + m_valid : len(X) + 1]
y_test = y[m_train + m_valid : len(y) + 1]

# if BATCH_LIMIT:
#     limit = BATCH_LIMIT*BATCH_SIZE
#     X_train = X_train[:limit, :]
#     Y_train = Y_train[:limit]

### Classification by distance from mean
print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_train.shape)
print("Data type:", X_train.dtype)

    # Training set predictions and score
    predictions_train = predict(X_train, label_means, distinct_labels)
    print("Training Set Accuracy:", sum(y_train == predictions_train)/m_train)

    # Test set predictions and score
    predictions_test = predict(X_test, label_means, distinct_labels)
    print("Test Set Accuracy:", sum(y_test == predictions_test)/m_test)
