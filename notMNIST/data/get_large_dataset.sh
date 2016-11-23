curl -O http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz
tar -xzf notMNIST_large.tar.gz
python data/matlab_convert.py notMNIST_large data/notMNIST_large.mat
rm notMNIST_large.tar.gz
rm -r notMNIST_large
# octave data/remove_duplicates.m data/notMNIST_large.mat
