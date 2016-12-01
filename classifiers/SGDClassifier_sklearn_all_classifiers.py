"""
Designed to compare all classifiers (loss functions) available through 
sklearn's SGDClassifier.
"""

# Standard Library Dependencies
from __future__ import division, print_function
from warnings import warn
from time import time
import os
from operator import itemgetter

# External Dependencies
import numpy as np
from scipy.io import loadmat
from numpy.linalg import norm
from sklearn.linear_model import SGDClassifier

all_loss_choices = ['hinge', 'log', 'modified_huber', 'squared_hinge', 
                    'perceptron', 'squared_loss', 'huber', 
                    'epsilon_insensitive', 'squared_epsilon_insensitive']

# User Parameters
DATASET = 'usps'
VALIDATION_PERCENTAGE = .2
TESTING_PERCENTAGE = .2
LOSS_FUNCTIONS = 'all'  # 'all' or a list of SGDClassifier loss functions
assert 0 < VALIDATION_PERCENTAGE + TESTING_PERCENTAGE < 1

if all:
    loss_choices = all_loss_choices
else:
    loss_choices = LOSS_FUNCTION

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

m_train = int(0.6*m)
X_train = X[:m_train]
y_train = y[:m_train]

m_valid = int(0.2*m)
X_valid = X[m_train : m_train + m_valid + 1]
y_valid = y[m_train : m_train + m_valid + 1]

m_test = m - m_valid - m_train
X_test = X[m_train + m_valid : len(X) + 1]
y_test = y[m_train + m_valid : len(y) + 1]


### Using sklearn and Stochastic Gradient Descent
print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_train.shape)
print("Data dtype:", X_train.dtype)

res = []
for loss_fcn in loss_choices:
    # Train
    tr_time = time()
    try:
        classifier = SGDClassifier(loss=loss_fcn).fit(X_train, y_train)
    except Exception as e:
        print(e)
        continue
    tr_time = time() - tr_time
    
    # Test
    te_time = time()
    # Z = classifier.predict(X_test)
    score = classifier.score(X_valid, y_valid)
    te_time = time() - te_time
    res.append((loss_fcn, score, tr_time, te_time))
    report = ("Method: {:>27} | Time: {:.3f}s | Accuracy: {:.2f}%"
              "".format(loss_fcn, tr_time + te_time, 100*score))
    print(report)

print('*'*50)
loss_fcn, score, tr_time, te_time = max(res, key=itemgetter(1))
print('Winner:', loss_fcn)
print("Training time:", tr_time)
# print("Testing time:", te_time)
print("~~Validation Accuracy: {}".format(score))
classifier = SGDClassifier(loss=loss_fcn).fit(X_train, y_train)
test_score = classifier.score(X_test, y_test)
print("~~Test Accuracy: {}\n".format(test_score))
