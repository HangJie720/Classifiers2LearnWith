# Standard Library Imports
from __future__ import division, print_function
from time import time

# External Imports
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
# %matplotlib inline  # use when in JUPYTER NOTEBOOK (or risk hang)
# plt.ion()  # allow ipython %run to terminate without closing figure

# Internal Imports
from utils import load_data

# User Parameters
DATASET = '/Users/Andy/Google Drive/Development/ML/kaggle/leaf-classification/KAGGLE_LEAF.mat'
# DATASET = 'usps'
VALIDATION_PERCENTAGE = .2
TESTING_PERCENTAGE = .2
# RANKS2TRY = range(10)
RANKS2TRY = 'all'  # must be 'all' or list of integers
TIMER_ON = False
assert 0 < VALIDATION_PERCENTAGE + TESTING_PERCENTAGE < 1

# Load dataset
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = \
    load_data(DATASET, VALIDATION_PERCENTAGE, TESTING_PERCENTAGE)

# only used for score, has no affect on approximation
batch_size = len(Y_train) // 10

# Classification by distance from best k-dimensional subspace 
# approximation from SVD of each label's example set
if RANKS2TRY == 'all':
    ranks2try = list(range(1, X_train.shape[1]))  # all must be less than full rank
else:
    ranks2try = RANKS2TRY
valid_accuracy = []
for rnk in ranks2try:
    start_time = time()
    print("")
    print("rank = {} -- ".format(rnk), end='')
    distinct_labels = list(set(Y_train))
    svd = {}
    for l in distinct_labels:
        examples_labeled_l = \
            np.array([x for x, y in zip(X_train, Y_train) if y == l])
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
            distances[l] = np.linalg.norm(X_ - X_appr, axis=1)
        distances = np.array(distances.values()).transpose()
        distance_minimizers = np.argmin(distances, axis=1)
        Y_predictions = [distinct_labels[idx] for idx in distance_minimizers]
        number_correct_ = np.sum(Y_predictions == Y_)
        return number_correct_

    batches = [(k*batch_size, (k+1)*batch_size) for k in
               range(len(Y_train) // batch_size)]
    ct = 0
    number_correct = 0
    for i0, i1 in batches:
        ct += 1
        number_correct_batch = \
            svd_predict(X_train[i0: i1], Y_train[i0: i1], svd)
        # print("Training Batch {}/{} Accuracy: {}"
        #       "".format(ct, len(batches), number_correct_batch/(i1 - i0)))
        number_correct += number_correct_batch
    if len(Y_train) % batch_size:
        i0, i1 = i1, len(Y_train)
        number_correct_batch = \
            svd_predict(X_train[i0: i1], Y_train[i0: i1], svd)
        # print("Training Remainder Batch Accuracy: {}"
        #       "".format(ct, len(batches), number_correct_batch/(i1 - i0)))
        number_correct += number_correct_batch
    print("Training / Validation Accuracy: {:.2f}% / "
          "".format(100 * number_correct / len(Y_train)), end='')

    # Validation Set Accuracy
    if batch_size < len(Y_valid):
        batches = [(k*batch_size, (k+1)*batch_size) for k in
                   range(len(Y_valid) // batch_size)]
        ct = 0
        number_correct = 0
        for i0, i1 in batches:
            ct += 1
            number_correct_batch = \
                svd_predict(X_valid[i0: i1], Y_valid[i0: i1], svd)
            # print("valid Batch {}/{} Accuracy: {}"
            #       "".format(ct, len(batches), 
                                # number_correct_batch/(i1 - i0)))
            number_correct += number_correct_batch
        if len(Y_valid) % batch_size:
            i0, i1 = i1, len(Y_valid)
            number_correct_batch = \
                svd_predict(X_valid[i0: i1], Y_valid[i0: i1], svd)
            # print("valid Remainder Batch Accuracy: {}"
            #       "".format(ct, len(batches), 
                                # number_correct_batch/(i1 - i0)))
            number_correct += number_correct_batch
    else:
        number_correct = svd_predict(X_valid, Y_valid, svd)
    if not number_correct/len(Y_valid):
        raise Exception()
    valid_accuracy.append(number_correct / len(Y_valid))
    print("{:.2f}%".format(100*valid_accuracy[-1]))
    if TIMER_ON:
        print("Time to Train and Validate with this rank: {:.2f} seconds"
              "".format(time() - start_time))
print("\nWinner winner chicken dinner goes to rank =",
      ranks2try[np.argmax(valid_accuracy)])

plt.grid(True)
plt.plot(ranks2try, valid_accuracy)

# Now that we've found the best rank to use.
rnk = ranks2try[np.argmax(valid_accuracy)]
distinct_labels = list(set(Y_train))
svd = {}
for l in distinct_labels:
    examples_labeled_l = np.array([x for x, y in
                                   zip(X_train, Y_train) if y == l])
    svd[l] = TruncatedSVD(n_components=rnk)
    svd[l].fit(examples_labeled_l)

# Test Set Accuracy
if batch_size < len(Y_test):
    batches = [(k*batch_size, (k+1)*batch_size) for k in
               range(len(Y_test) // batch_size)]
    ct = 0
    number_correct = 0
    for i0, i1 in batches:
        ct += 1
        number_correct_batch = \
            svd_predict(X_test[i0: i1], Y_test[i0: i1], svd)
        # print("Test Batch {}/{} Accuracy: {}"
        #     "".format(ct, len(batches), number_correct_batch/(i1 - i0)))
        number_correct += number_correct_batch
    if len(Y_test) % batch_size:
        i0, i1 = i1, len(Y_test)
        number_correct_batch = \
            svd_predict(X_test[i0: i1], Y_test[i0: i1], svd)
        # print("Test Remainder Batch Accuracy: {}"
        #       "".format(ct, len(batches), number_correct_batch/(i1 - i0))) 
        number_correct += number_correct_batch
else:
    number_correct = svd_predict(X_test, Y_test, svd)
print("Test Accuracy with winner: {:.2f}%"
      "".format(100 * number_correct / len(Y_test)))

plt.show()  # prevent python from terminating and closing figure