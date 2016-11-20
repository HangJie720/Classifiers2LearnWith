from load_mnist import load_mnist
X_train, y_train = load_mnist('training')
X_test, y_test = load_mnist('testing')

from scipy.io import savemat
mnist_dict = {'X_train' : X_train, 'y_train' : y_train,
              'X_test' : X_test, 'y_test' : y_test}
savemat('MNIST.mat', mnist_dict)