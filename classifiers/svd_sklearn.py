# Standard Library Imports
from __future__ import division, print_function
from warnings import warn
from time import time
import os

# External Imports
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import norm
norm2 = np.linalg.norm
import matplotlib.pyplot as plt
# %matplotlib inline  # use when in JUPYTER NOTEBOOK (or risk hang)
# plt.ion()  # allow ipython %run to terminate without closing figure

# User Parameters
DATASET = 'usps'
VALIDATION_PERCENTAGE = .2
TESTING_PERCENTAGE = .2
RANKS2TRY = 'all'  # must be 'all' or list of integers
assert 0 < VALIDATION_PERCENTAGE + TESTING_PERCENTAGE < 1

# Load dataset
rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

X = dataset_dict['X']
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

# normalize data
# X = X - mean(X);  % hurts accuracy
# X = X./mean(sum((X-mean(X)).^2));  % doesn't help accuracy

m_train = int((1-TESTING_PERCENTAGE-VALIDATION_PERCENTAGE)*m)
X_train = X[:m_train]
y_train = y[:m_train]

m_valid = int(VALIDATION_PERCENTAGE*m)
X_valid = X[m_train : m_train + m_valid + 1]
y_valid = y[m_train : m_train + m_valid + 1]

m_test = m - m_valid - m_train
X_test = X[m_train + m_valid : len(X) + 1]
y_test = y[m_train + m_valid : len(y) + 1]

# only used for score, has no affect on approximation
batch_size = len(y_train)//10  

# Classification by distance from best k-dimensional subspace 
# approximation from SVD of each label's example set
if RANKS2TRY == 'all':
    ranks2try = list(range(1, n))  # all must be less than full rank
else:
    ranks2try = RANKS2TRY
valid_accuracy = []
for rnk in ranks2try:
    start_time = time()
    print("")
    print("rank =", rnk)
    distinct_labels = list(set(y_train))
    svd = {}
    for l in distinct_labels:
        examples_labeled_l = \
            np.array([x for x, y in zip(X_train, y_train) if y == l])
        svd[l] = TruncatedSVD(n_components=rnk)
        svd[l].fit(examples_labeled_l)

    # Training Set Accuracy
    def svd_predict(X_, Y_, svd_dict_):
        X_, Y_ = np.array(X_), np.array(Y_)
        distinct_labels = svd_dict_.keys()
        distances = {}
        for l in distinct_labels:
            X_appr = svd_dict_[l].inverse_transform(
                                    svd_dict_[l].transform(X_))
            distances[l] = norm2(X_ - X_appr, axis=1)
        distances = np.array(distances.values()).transpose()
        distance_minimizers = np.argmin(distances, axis=1)
        Y_predictions = [distinct_labels[idx] for idx in distance_minimizers]
        number_correct_ = np.sum(Y_predictions == Y_)
        return number_correct_

    batches = [(k*batch_size, (k+1)*batch_size) for k in 
                                            range(len(y_train)//batch_size)]
    ct = 0
    number_correct = 0
    for i0, i1 in batches:
        ct += 1
        number_correct_batch = \
            svd_predict(X_train[i0: i1], y_train[i0: i1], svd)
        # print("Training Batch {}/{} Accuracy: {}"
        #       "".format(ct, len(batches), number_correct_batch/(i1 - i0)))
        number_correct += number_correct_batch
    if len(y_train) % batch_size:
        i0, i1 = i1, len(y_train)
        number_correct_batch = \
            svd_predict(X_train[i0: i1], y_train[i0: i1], svd)
        # print("Training Remainder Batch Accuracy: {}"
        #       "".format(ct, len(batches), number_correct_batch/(i1 - i0)))
        number_correct += number_correct_batch
    print("Training Accuracy: {:.2f}%".format(100*number_correct/len(y_train)))

    # Validation Set Accuracy
    if batch_size < len(y_valid):
        batches = [(k*batch_size, (k+1)*batch_size) for k in 
                                            range(len(y_valid)//batch_size)]
        ct = 0
        number_correct = 0
        for i0, i1 in batches:
            ct += 1
            number_correct_batch = \
                svd_predict(X_valid[i0: i1], y_valid[i0: i1], svd)
            # print("valid Batch {}/{} Accuracy: {}"
            #       "".format(ct, len(batches), 
                                # number_correct_batch/(i1 - i0)))
            number_correct += number_correct_batch
        if len(y_valid) % batch_size:
            i0, i1 = i1, len(y_valid)
            number_correct_batch = \
                svd_predict(X_valid[i0: i1], y_valid[i0: i1], svd)
            # print("valid Remainder Batch Accuracy: {}"
            #       "".format(ct, len(batches), 
                                # number_correct_batch/(i1 - i0)))
            number_correct += number_correct_batch
    else:
        number_correct = svd_predict(X_valid, y_valid, svd)
    if not number_correct/len(y_valid):
        raise Exception()
    valid_accuracy.append(number_correct/len(y_valid))
    print("Validation Set Accuracy: {:.2f}%".format(100*valid_accuracy[-1]))
    print("Time to Train and Validate with this rank: {:.2f} seconds"
          "".format(time() - start_time))
print("\nWinner winner chicken dinner goes to rank =", 
        100*ranks2try[np.argmax(valid_accuracy)])

plt.grid(True)
plt.plot(ranks2try, valid_accuracy)

# Now that we've found the best rank to use.
from sklearn.decomposition import TruncatedSVD
norm2 = np.linalg.norm

rnk = ranks2try[np.argmax(valid_accuracy)]
distinct_labels = list(set(y_train))
svd = {}
for l in distinct_labels:
    examples_labeled_l = np.array([x for x, y in 
                                    zip(X_train, y_train) if y == l])
    svd[l] = TruncatedSVD(n_components=rnk)
    svd[l].fit(examples_labeled_l)

# Test Set Accuracy
if batch_size < len(y_test):
    batches = [(k*batch_size, (k+1)*batch_size) for k in 
                                            range(len(y_test)//batch_size)]
    ct = 0
    number_correct = 0
    for i0, i1 in batches:
        ct += 1
        number_correct_batch = \
            svd_predict(X_test[i0: i1], y_test[i0: i1], svd)
        # print("Test Batch {}/{} Accuracy: {}"
        #     "".format(ct, len(batches), number_correct_batch/(i1 - i0)))
        number_correct += number_correct_batch
    if len(y_test) % batch_size:
        i0, i1 = i1, len(y_test)
        number_correct_batch = \
            svd_predict(X_test[i0: i1], y_test[i0: i1], svd)
        # print("Test Remainder Batch Accuracy: {}"
        #       "".format(ct, len(batches), number_correct_batch/(i1 - i0))) 
        number_correct += number_correct_batch
else:
    number_correct = svd_predict(X_test, y_test, svd)
print("Test Accuracy with winner: {:.2f}%"
      "".format(100*number_correct/len(y_test)))

plt.show()  # prevent python from terminating and closing figure