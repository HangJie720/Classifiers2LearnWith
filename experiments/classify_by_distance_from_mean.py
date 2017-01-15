"""
Classify by distance from mean
"""

# Standard Library Dependencies
from __future__ import division, print_function, absolute_import

# Internal Dependencies
from utils import load_data
from classifiers import DistanceFromMeanClassifier

# User Parameters
DATASET = 'USPS'
VALIDATION_PERCENTAGE = 0
TESTING_PERCENTAGE = .3
NORMALIZE = False


X_train, Y_train, X_valid, Y_valid, X_test, Y_test = \
    load_data(DATASET, VALIDATION_PERCENTAGE, TESTING_PERCENTAGE,
              force_index_friendly_labels=True)


print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_train.shape)
print("Data type:", X_train.dtype)


classifier = DistanceFromMeanClassifier()
# Training set predictions and score
classifier.train(X_train, Y_train)
predictions_train = classifier.predict(X_train)
print("Training Set Accuracy:", sum(Y_train == predictions_train) / len(Y_train))


# Test set predictions and score
predictions_test = classifier.predict(X_test)
print("Test Set Accuracy:", sum(Y_test == predictions_test) / len(Y_test))
