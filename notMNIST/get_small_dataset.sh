curl -O http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
tar -xzf notMNIST_small.tar.gz
python matlab_convert.py notMNIST_small data/notMNIST_small.mat
rm notMNIST_small.tar.gz
rm -r notMNIST_small
