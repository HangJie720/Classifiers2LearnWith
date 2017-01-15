"""
Multinomial Logistic Regression using TensorFlow, no regularization.
"""

# Standard Library Dependencies
from __future__ import division, print_function
from warnings import warn
from time import time
import os
from operator import itemgetter

# External Dependencies
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline  # use when in JUPYTER NOTEBOOK (or risk hang)
# plt.ion()  # allow ipython %run to terminate without closing figure

# Internal Dependencies
from utils import load_data


# User Parameters
# DATASET = '/Users/Andy/Google Drive/Development/ML/kaggle/leaf-classification/KAGGLE_LEAF.mat'
DATASET = 'usps'
VALIDATION_PERCENTAGE = .2
TESTING_PERCENTAGE = .2
DESCENT_STEPS = 1000
NORMALIZE = False
BATCH_SIZE = 128
USE_NONSTOCHASTIC_GD = True  # just for fun, slow...
assert 0 < VALIDATION_PERCENTAGE + TESTING_PERCENTAGE < 1


# Load dataset
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = \
    load_data(DATASET, VALIDATION_PERCENTAGE, TESTING_PERCENTAGE, one_hot=True)


print("Training Set Shape:", X_train.shape, Y_train.shape)
print("Validation Set Shape:", X_valid.shape, Y_valid.shape)
print("Testing Set Shape:", X_test.shape, Y_test.shape)
print("Data type:", X_train.dtype)


m_train, n = X_train.shape
number_of_classes = len(Y_train[0])
if BATCH_SIZE == 'full':
    batch_size = X_train.shape[0]
else:
    batch_size = BATCH_SIZE


def accuracy(predictions, labels):
    is_correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    return 100.0 * tf.reduce_mean(tf.cast(is_correct, "float"))

graph = tf.Graph()
with graph.as_default():
    # Specify shape of input
    if USE_NONSTOCHASTIC_GD:
        X = tf.constant(X_train, dtype=tf.float32)
        Y = tf.constant(Y_train, dtype=tf.float32)
    else:
        X = tf.placeholder(tf.float32, shape=(batch_size, X_train.shape[1]))
        Y = tf.placeholder(tf.float32, shape=(batch_size, number_of_classes))

    # Specify variables
    weights = tf.Variable(tf.truncated_normal([n, number_of_classes]))
    biases = tf.Variable(tf.zeros([number_of_classes]))

    # Training
    logits = tf.matmul(X, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, Y))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Make Predictions
    X_valid = tf.constant(X_valid, dtype=tf.float32)
    Y_valid = tf.constant(Y_valid, dtype=tf.float32)
    X_test = tf.constant(X_test, dtype=tf.float32)

    # train_prediction = tf.nn.softmax(logits)
    train_accuracy = accuracy(tf.nn.softmax(logits), Y)
    valid_logits = tf.matmul(X_valid, weights) + biases
    valid_accuracy = accuracy(tf.nn.softmax(valid_logits), Y_valid)
    valid_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(valid_logits, Y_valid))

    test_prediction = tf.nn.softmax(tf.matmul(X_test, weights) + biases)
    test_accuracy = accuracy(test_prediction, Y_test)


results = []
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(DESCENT_STEPS):
        if USE_NONSTOCHASTIC_GD:
            _ = session.run([optimizer])
        else:
            offset = (step * BATCH_SIZE) % (Y_train.shape[0] - BATCH_SIZE)
            X_batch = X_train[offset:(offset + BATCH_SIZE), :]
            Y_batch = Y_train[offset:(offset + BATCH_SIZE), :]

            feed_dict = {X: X_batch, Y: Y_batch}
            _ = session.run([optimizer], feed_dict=feed_dict)

        if (step % 100) == 0:
            results.append((step, valid_loss.eval()))
            print('Loss at step {}: {}  ||  '
                  ''.format(results[-1][0], results[-1][1]), end='')
            print('Training score: {:.2f}%  ||  '
                  ''.format(train_accuracy.eval()), end='')
            # valid_score = accuracy(valid_prediction.eval(), Y_valid)
            print('Validation score: {:.2f}%'.format(valid_accuracy.eval()))
    print('Test score: {:.2f}%'.format(test_accuracy.eval()))

plt.grid(True)
plt.plot(*zip(*results))
plt.show()
