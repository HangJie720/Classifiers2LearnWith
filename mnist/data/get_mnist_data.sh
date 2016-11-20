#!/bin/bash
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
7z x train-images-idx3-ubyte.gz
7z x train-labels-idx1-ubyte.gz
7z x t10k-images-idx3-ubyte.gz
7z x t10k-labels-idx1-ubyte.gz
rm *ubyte.gz
python matlab_convert.py
rm *ubyte
