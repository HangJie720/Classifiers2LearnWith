"""
Fully Connected Single Hidden Layer Network using TensorFlow.
"""

# Standard Library Dependencies
from __future__ import division, print_function
from warnings import warn
from time import time
import os
from operator import itemgetter

# External Dependencies
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from numpy.linalg import norm
import matplotlib.pyplot as plt
# %matplotlib inline  # use when in JUPYTER NOTEBOOK (or risk hang)
# plt.ion()  # allow ipython %run to terminate without closing figure

# User Parameters
DATASET = 'usps'
VALIDATION_PERCENTAGE = .2
TESTING_PERCENTAGE = .2
DESCENT_STEPS = 500
TESTING_DESCENT_STEPS = 3000
NORMALIZE = False
HIDDEN_LAYER_NODES = 1024
BATCH_SIZE = 128  # set to zero to use non-stochastic optimizer
DROPOUT = .5
REG_COEFFS2TRY = [10**k for k in np.arange(-4, -2, 0.1)]
LEARNING_RATE = 0.5
BATCH_LIMIT = 0  # Limit number of batches used for training

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
def onehot(data, ordered_label_set):
    return [[(1 if l == y else 0) for l in ordered_label_set] for y in data]
y = np.array(onehot(y, distinct_labels)).astype('float32')
print(y.shape)

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

if BATCH_LIMIT:
    limit = BATCH_LIMIT*BATCH_SIZE
    X_train = X_train[:limit, :]
    y_train = y_train[:limit]

### TensorFlow 1-layer fully_connected with (non-stochastic) gradient descent
print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_train.shape)
print("Data type:", X_train.dtype)

N_in = n
N_1 = HIDDEN_LAYER_NODES
N_out = len(distinct_labels)

graph = tf.Graph()
with graph.as_default():

    # Inputs to be fed in at each step
    X = tf.placeholder(tf.float32,shape=(BATCH_SIZE, N_in))
    Y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, N_out))
    reg_coeff = tf.placeholder('float')

    # hidden layer
    weights_1 = tf.Variable(tf.truncated_normal([N_in, N_1]))
    biases_1 = tf.Variable(tf.zeros([N_1]))
    a1 = tf.nn.relu(tf.matmul(X, weights_1) + biases_1)
    
    # output layer
    weights_f = tf.Variable(tf.truncated_normal([N_1, N_out]))
    biases_f = tf.Variable(tf.zeros([N_out]))
    logits = tf.matmul(a1, weights_f) + biases_f
    
    # Add dropout
    a1_d = tf.nn.dropout(a1, DROPOUT)
    logits_d = tf.matmul(a1_d, weights_f) + biases_f

    # Training Method
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits_d, Y)
    regul = reg_coeff*(tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_f))

    loss = tf.reduce_mean(cross_entropy)  + regul
    beta = [weights_1, biases_1, weights_f, biases_f]
    optimizer = tf.train.GradientDescentOptimizer(
                    LEARNING_RATE).minimize(loss, var_list=beta)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
        
    # Setup Validation (to be compared after training with y_valid)
    X_valid = tf.constant(X_valid)
    a1 = tf.matmul(X_valid, weights_1) + biases_1
    a2 = tf.matmul(tf.nn.relu(a1), weights_f) + biases_f
    valid_prediction = tf.nn.softmax(a2)
    
    # Setup Testing (to be compared after training with y_test)
    X_test = tf.constant(X_test)
    a1 = tf.matmul(X_test, weights_1) + biases_1
    a2 = tf.matmul(tf.nn.relu(a1), weights_f) + biases_f
    test_prediction = tf.nn.softmax(a2)


def accuracy(predictions, labels):
    num_correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    return 100.0 * num_correct / predictions.shape[0]

validation_scores = []
for reg in REG_COEFFS2TRY:
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('-'*50)
        print('Training with Regularization Coefficient: ', reg)
        for step in range(DESCENT_STEPS):
            feed_dict = {}

            # get batch ready for training step
            if BATCH_SIZE:
                offset = (step*BATCH_SIZE) % (y_train.shape[0] - BATCH_SIZE)
                X_batch = X_train[offset:(offset + BATCH_SIZE), :]
                y_batch = y_train[offset:(offset + BATCH_SIZE), :]
            else:
                X_batch = X_train
                y_batch = y_train

            # Training Step
            feed_dict = {X : X_batch, Y : y_batch, reg_coeff : reg}
            fetches = [optimizer, loss, train_prediction]
            if feed_dict:
                _, l, predictions = session.run(fetches, feed_dict=feed_dict)
            else:
                _, l, predictions = session.run(fetches)

            if (step % 100 == 0):
                train_accuracy = accuracy(predictions, y_batch)
                print('Step: {} | Loss: {} | Training Accuracy: {:.2f}%'
                      ''.format(step, l, train_accuracy))
        # Validate
        valid_accuracy = accuracy(valid_prediction.eval(), y_valid)
        if not validation_scores or valid_accuracy > max(validation_scores):
            best_test_accuracy = accuracy(test_prediction.eval(), y_test)
            best_valid_accuracy = valid_accuracy
            best_reg_coeff = reg
        print('Validation Accuracy: {:.2f}%'.format(valid_accuracy))
        validation_scores.append(valid_accuracy)

print('*'*50)
print('Winner: reg_coeff', best_reg_coeff)
print("~~Validation Accuracy: {:.2f}%".format(best_valid_accuracy))

if TESTING_DESCENT_STEPS > DESCENT_STEPS:
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(TESTING_DESCENT_STEPS):
            feed_dict = {}

            # get batch ready for training step
            if BATCH_SIZE:
                offset = (step*BATCH_SIZE) % (y_train.shape[0] - BATCH_SIZE)
                X_batch = X_train[offset:(offset + BATCH_SIZE), :]
                y_batch = y_train[offset:(offset + BATCH_SIZE), :]
            else:
                X_batch = X_train
                y_batch = y_train

            # Training Step
            feed_dict = {X : X_batch, Y : y_batch, reg_coeff : reg}
            fetches = [optimizer, loss, train_prediction]
            if feed_dict:
                _, l, predictions = session.run(fetches, feed_dict=feed_dict)
            else:
                _, l, predictions = session.run(fetches)

            if (step % 100 == 0):
                train_accuracy = accuracy(predictions, y_batch)
                print('Step: {} | Loss: {} | Training Accuracy: {:.2f}%'
                      ''.format(step, l, train_accuracy))

        # Get Test Score
        test_accuracy = accuracy(valid_prediction.eval(), y_valid)
    print('*'*50)
    print('Winner: reg_coeff', best_reg_coeff)
    print("~~Validation Accuracy: {:.2f}%".format(best_valid_accuracy))
print("~~Test Accuracy: {:.2f}%".format(best_test_accuracy))

plt.semilogx(REG_COEFFS2TRY, validation_scores)
plt.grid(True)
plt.title('regularization coeff vs validation accuracy\n'
          'best coeff: {}  accuracy: {}'
          ''.format(best_reg_coeff, best_valid_accuracy))
plt.show()
