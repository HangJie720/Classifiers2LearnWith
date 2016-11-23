To use these examples, you must download/extract/convert the datasets
To aquire the small (sample ~100mb) and large (full > 3gb) notMNIST datasets,
use the respective scripts: "get_small_dataset.sh" and "get_large_dataset.sh"

Easy Way:=======================================
$ sh data/get_small_dataset.sh
$ sh data/get_large_dataset.sh

# (Optional:) Use data/remove_duplicates.m to create *_unique.mat files
$ octave data/remove_duplicates.m data/notMNIST_small.mat
$ octave data/remove_duplicates.m data/notMNIST_large.mat



Slightly Less Easy Way:=========================
To get the full dataset notMNIST_large.mat (small similar):
1) download the compressed dataset at 
$ http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz

2) decompress
$ tar -xzf notMNIST_large.tar.gz

3) convert to .mat 
$ python data/matlab_convert.py notMNIST_large data/notMNIST_large.mat

4) delete the now unneeded notMNIST_large directory and tar.gz file

5) (Optional:) Use data/remove_duplicates.m to create *_unique.mat files
$ octave data/remove_duplicates.m data/notMNIST_small.mat
$ octave data/remove_duplicates.m data/notMNIST_large.mat