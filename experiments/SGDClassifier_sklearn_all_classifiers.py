"""
Designed to compare all classifiers (loss functions) available through 
sklearn's SGDClassifier.
"""

# Standard Library Dependencies
from __future__ import division, print_function
from time import time
import os
from operator import itemgetter

# Internal Dependencies
from utils import diagnostic, load_data, printmat

# External Dependencies
from scipy.io import loadmat
from sklearn.linear_model import SGDClassifier

all_loss_choices = ['hinge', 'log', 'modified_huber', 'squared_hinge', 
                    'perceptron', 'squared_loss', 'huber', 
                    'epsilon_insensitive', 'squared_epsilon_insensitive']

# User Parameters
DATASET = '/Users/Andy/Google Drive/Development/ML/kaggle/leaf-classification/KAGGLE_LEAF.mat'
# DATASET = 'usps'
VALIDATION_PERCENTAGE = .2
TESTING_PERCENTAGE = .2
LOSS_FUNCTIONS = 'all'  # 'all' or a list of SGDClassifier loss functions
assert 0 < VALIDATION_PERCENTAGE + TESTING_PERCENTAGE < 1


# Parse User Parameters
if LOSS_FUNCTIONS == 'all':
    loss_choices = all_loss_choices
else:
    loss_choices = LOSS_FUNCTIONS

# Load dataset
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = \
    load_data(DATASET, VALIDATION_PERCENTAGE, TESTING_PERCENTAGE)

# Using sklearn and Stochastic Gradient Descent
print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_train.shape)
print("Data dtype:", X_train.dtype)

valid_diag = diagnostic(Y_valid)
res = []
for loss_fcn in loss_choices:
    # Train
    tr_time = time()
    try:
        classifier = SGDClassifier(loss=loss_fcn).fit(X_train, Y_train)
    except Exception as e:
        print(e)
        continue
    tr_time = time() - tr_time
    
    # Test
    te_time = time()
    # Z = classifier.predict(X_test)
    score = classifier.score(X_valid, Y_valid)
    te_time = time() - te_time
    res.append((loss_fcn, score, tr_time, te_time))
    report = ("Method: {:>27} | Time: {:.3f}s | Accuracy: {:.2f}%"
              "".format(loss_fcn, tr_time + te_time, 100*score))
    print(report)
    valid_diag.diagnose(classifier.predict(X_valid), method=loss_fcn)

print('*'*50)
loss_fcn, score, tr_time, te_time = max(res, key=itemgetter(1))
print('Winner:', loss_fcn)
print("Training time:", tr_time)
# print("Testing time:", te_time)
print("~~Validation Accuracy: {}".format(score))
classifier = SGDClassifier(loss=loss_fcn).fit(X_train, Y_train)
test_score = classifier.score(X_test, Y_test)
print("~~Test Accuracy: {}\n".format(test_score))

print('*'*50)
print("Diagnstic:")
valid_diag.report()
