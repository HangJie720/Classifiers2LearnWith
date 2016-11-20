To use these examples, you must download/extract/convert the datasets
To aquire the small (sample ~100mb) and large (full > 3gb) notMNIST datasets,
use the respective scripts: "get_small_dataset.sh" and "get_large_dataset.sh"

Easy Way:=======================================
$ sh get_small_dataset.sh
$ sh get_large_dataset.sh


Slightly Less Easy Way:=========================
To get the full dataset notMNIST_large.mat (small similar):
1) download the compressed dataset at 
$ http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz

2) decompress
$ tar -xzf notMNIST_large.tar.gz

3) convert to .mat 
$ python matlab_convert.py notMNIST_large data/notMNIST_large.mat