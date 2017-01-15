"""
Fully Connected Network using TensorFlow.

Note on running (nec. b/c of utils import):
be in root project folder, ipython2, THEN cd experiments
"""

# Standard Library Dependencies
from __future__ import division, print_function, absolute_import

# External Dependencies
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline  # use when in JUPYTER NOTEBOOK (or risk hang)
# plt.ion()  # allow ipython %run to terminate without closing figure

# Internal Dependencies
from utils import load_data

# User Parameters
DATASET = 'usps'
VALIDATION_PERCENTAGE = .2
TESTING_PERCENTAGE = .2
DESCENT_STEPS = 500
TESTING_DESCENT_STEPS = 3000
NORMALIZE = False
HIDDEN_LAYER_SIZES = [1024, 2048]
BATCH_SIZE = 128
DROPOUT = .5
# REGULARIZATION = [10**k for k in np.arange(-4, -2, 0.1)]  # Use [0] for no regularization
REGULARIZATION = [0.2, 0.3]
LEARNING_RATE = 0.5
BATCH_LIMIT = 0  # Limit number of batches used for training

assert 0 < VALIDATION_PERCENTAGE + TESTING_PERCENTAGE < 1

# Load dataset
data = load_data(DATASET, VALIDATION_PERCENTAGE, TESTING_PERCENTAGE, one_hot=True)
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = [np.array(x, dtype='float32') for x in data]

if BATCH_LIMIT:
    limit = BATCH_LIMIT*BATCH_SIZE
    X_train = X_train[:limit, :]
    Y_train = Y_train[:limit]

print("Data shape:", *[x.shape for x in data])
print("Data dtype:", *[x.dtype for x in data])


def accuracy(predictions, labels):
    is_correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    return 100.0 * tf.reduce_mean(tf.cast(is_correct, "float"))


N_in = X_train.shape[1]
N_out = Y_train.shape[1]
N = [N_in] + HIDDEN_LAYER_SIZES + [N_out]

graph = tf.Graph()
with graph.as_default():
    gradient_descent = tf.train.GradientDescentOptimizer(LEARNING_RATE)

    # Inputs to be fed in at each step
    X = tf.placeholder(X_train.dtype, shape=(BATCH_SIZE, N_in), name='X')
    Y = tf.placeholder(Y_train.dtype, shape=(BATCH_SIZE, N_out), name='Y')
    reg_coeff = tf.placeholder('float32', name='regularization_coefficient')

    # hidden layer
    W = []; b = []; a = X
    for n_in, n_out in zip(N[:-1], N[1:]):
        W.append(tf.Variable(tf.truncated_normal((n_in, n_out)), dtype='float32'))
        b.append(tf.Variable(tf.zeros((n_out,)), dtype='float32'))
        logits = tf.matmul(a, W[-1]) + b[-1]
        if len(W) != len(N)-1:
            if DROPOUT:
                a = tf.nn.dropout(tf.nn.relu(logits), DROPOUT)
            else:
                a = tf.nn.relu(logits)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y)

    if REGULARIZATION == [0]:
        reg_term = 0
    else:
        reg_term = reg_coeff * sum(tf.nn.l2_loss(Wl) for Wl in W)

    loss = tf.reduce_mean(cross_entropy) + reg_term
    optimizer = gradient_descent.minimize(loss, var_list=W+b)

    # Predictions for the training, validation, and test data
    def feed_forward_(a_, W_, b_):
        for Wl_, bl_ in zip(W_, b_)[:-1]:
            a_ = tf.nn.relu(tf.matmul(a_, Wl_) + bl_)
        return tf.nn.softmax(tf.matmul(a_, W_[-1]) + b_[-1])

    train_prediction = feed_forward_(X, W, b)  # can't use eval for this as X is placeholder
    train_accuracy = accuracy(train_prediction, Y)  # can't use eval for this as X is placeholder

    tr_X_valid = tf.constant(X_valid)
    tr_Y_valid = tf.constant(Y_valid)
    valid_prediction = feed_forward_(tr_X_valid, W, b)
    valid_accuracy = accuracy(valid_prediction, tr_Y_valid)

    tr_X_test = tf.constant(X_test)
    tr_Y_test = tf.constant(Y_test)
    test_prediction = feed_forward_(tr_X_test, W, b)
    test_accuracy = accuracy(test_prediction, tr_Y_test)


best_valid_accuracy = 0
best_reg_coeff = REGULARIZATION[0]
validation_scores = []
if len(REGULARIZATION) > 1:
    # try:  # In case of KeyboardInterrupt, show plots
    for reg in REGULARIZATION:
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()

            print('-' * 50)
            print('Training with Regularization Coefficient: ', reg)
            for step in range(DESCENT_STEPS):

                # get batch ready for training step
                if BATCH_SIZE:
                    offset = (step * BATCH_SIZE) % (Y_train.shape[0]-BATCH_SIZE)
                    X_batch = X_train[offset:(offset + BATCH_SIZE), :]
                    Y_batch = Y_train[offset:(offset + BATCH_SIZE), :]
                else:
                    X_batch = X_train
                    Y_batch = Y_train

                # Training Step
                feed_dict = {X: X_batch, Y: Y_batch, reg_coeff: reg}
                if step % 100:
                    fetches = [optimizer, loss]
                    _, training_loss = session.run(fetches, feed_dict=feed_dict)
                else:
                    fetches = [optimizer, loss, train_accuracy]
                    _, training_loss, train_accuracy_ = \
                        session.run(fetches, feed_dict=feed_dict)
                    print('Step: {} | Loss: {} | Training Accuracy: {:.2f}%'
                          ''.format(step, training_loss, train_accuracy_))
            # Validate
            valid_accuracy_ = valid_accuracy.eval()
            if valid_accuracy_ > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy_
                best_reg_coeff = reg

            print('Validation score: {:.2f}%'.format(valid_accuracy_))
            validation_scores.append(valid_accuracy_)

    # except KeyboardInterrupt as e:
    #     print(e)
    # finally:
    print('*' * 50)
    print('Winner: reg_coeff', best_reg_coeff)
    print("~~Validation Accuracy: {:.2f}%".format(best_valid_accuracy))
    plt.semilogx(REGULARIZATION, validation_scores)
    plt.grid(True)
    plt.title('regularization coeff vs validation score\n'
              'best coeff: {}  score: {}'
              ''.format(best_reg_coeff, best_valid_accuracy))


# Train further with best regularization coefficient, including validation data
X_train = np.vstack((X_train, X_valid))
Y_train = np.vstack((Y_train, Y_valid))
test_scores = []
# try:  # In case of KeyboardInterrupt, pickle training progress and show plots
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(TESTING_DESCENT_STEPS):

        # get batch ready for training step
        if BATCH_SIZE:
            offset = (step * BATCH_SIZE) % (Y_train.shape[0] - BATCH_SIZE)
            X_batch = X_train[offset:(offset + BATCH_SIZE), :]
            Y_batch = Y_train[offset:(offset + BATCH_SIZE), :]
        else:
            X_batch = X_train
            Y_batch = Y_train

        # Training Step
        feed_dict = {X: X_batch, Y: Y_batch, reg_coeff: best_reg_coeff}
        fetches = [optimizer, loss]
        _, training_loss = session.run(fetches, feed_dict=feed_dict)

        if (step % 1000) == 0:
            test_accuracy_ = test_accuracy.eval()
            print('Step: {} | Loss: {} | Test Set Accuracy: {:.2f}%'
                  ''.format(step, training_loss, test_accuracy_))
            test_scores.append((step, test_accuracy_))

# except KeyboardInterrupt:
if validation_scores:
    plt.semilogx(REGULARIZATION, validation_scores)
    plt.grid(True)
    plt.title('regularization coeff vs validation score\n'
              'best coeff: {}  score: {}'
              ''.format(best_reg_coeff, best_valid_accuracy))
    plt.show()

plt.plot(*zip(*test_scores))
plt.grid(True)
plt.title('SGD step vs test score')
plt.show()
