#!/bin/bash
hopefullyNOoneUSESthisNAME="$DATASET"
DATASET=notMNIST_small
curl -O http://yaroslavvb.com/upload/notMNIST/"$DATASET".tar.gz
tar -xzf "$DATASET".tar.gz
python data/notMNIST/matlab_convert.py "$DATASET" data/notMNIST/"$DATASET".mat
rm "$DATASET".tar.gz
rm -r "$DATASET"
octave data/notMNIST/remove_duplicates.m data/notMNIST/"$DATASET".mat
# rm data/notMNIST/"$DATASET".mat
DATASET="$hopefullyNOoneUSESthisNAME"
