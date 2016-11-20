Dataset downloaded from:
http://yann.lecun.com/exdb/mnist/


To load in python use included load_mnist.py
###############################################################
# Python code #################################################
# load data
from load_mnist import load_mnist
X_train, y_train = load_mnist('training')
X_test, y_test = load_mnist('testing')

# Now for example
import pylab
pylab.imshow(X_train[0], cmap=pylab.cm.gray)
pylab.show()

# end of Python code ##########################################
###############################################################


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
EXPLANATION of THE IDX FILE TYPE, from Lecun:
the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
The basic format is

magic number 
size in dimension 0 
size in dimension 1 
size in dimension 2 
..... 
size in dimension N 
data

The magic number is an integer (MSB first). The first 2 bytes are always 0.

The third byte codes the type of the data: 
0x08: unsigned byte 
0x09: signed byte 
0x0B: short (2 bytes) 
0x0C: int (4 bytes) 
0x0D: float (4 bytes) 
0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....

The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

The data is stored like in a C array, i.e. the index in the last dimension changes the fastest. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
