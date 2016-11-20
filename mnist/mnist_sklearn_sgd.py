from __future__ import print_function
import numpy as np
from scipy.io import loadmat
from numpy.linalg import norm
from warnings import warn
from time import time

# Load data
mnist_dict = loadmat("data/MNIST.mat")

def shape_data(X, y):
    m, w, h = X.shape
    return X.reshape((m, w*h)), y.reshape(m)
X_train, y_train = shape_data(mnist_dict['X_train'], mnist_dict['y_train'])
X_test, y_test = shape_data(mnist_dict['X_test'], mnist_dict['y_test'])


# shuffle data
from random import shuffle
def shuffle_data(X, y):
    p = np.random.permutation(len(y))
    return X[p], y[p]
X_train, y_train = shuffle_data(X_train, y_train)
X_test, y_test = shuffle_data(X_test, y_test)


# normalize data
# X = X - mean(X);  % hurts accuracy
# X = X./mean(sum((X-mean(X)).^2));  % doesn't help accuracy


# Take some training examples for a validation set
X_valid = X_train[:10**4]
y_valid = y_train[:10**4]
X_train = X_train[10**4:]
y_train = y_train[10**4:]


### Using sklearn and Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
loss_choices = ['hinge', 'log', 'modified_huber', 'squared_hinge', 
                'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 
                'squared_epsilon_insensitive']


print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_train.shape)
print("Data dtype:", X_train.dtype)

res = []
for loss_fcn in loss_choices:
    print("="*50)
    print("Method:", loss_fcn)
    
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
    print("Training time:", tr_time)
    # print("Testing time:", te_time)
    print("~~Validation Accuracy: {}\n".format(score))
    res.append((loss_fcn, score, tr_time, te_time))

print('*'*50)
from operator import itemgetter
loss_fcn, score, tr_time, te_time = max(res, key=itemgetter(1))
print('Winner:', loss_fcn)
print("Training time:", tr_time)
# print("Testing time:", te_time)
print("~~Validation Accuracy: {}\n".format(score))
classifier = SGDClassifier(loss=loss_fcn).fit(X_train, y_train)
test_score = classifier.score(X_test, y_test)
print("~~Test Accuracy: {}\n".format(test_score))
classifier = SGDClassifier(loss=loss_fcn).fit(np.vstack((X_train, X_valid)), 
                                            np.concatenate((y_train, y_valid)))
combined_test_score = classifier.score(X_test, y_test)
print("~~Test Accuracy (when training with training+validation set): {}\n"
    "".format(combined_test_score))

