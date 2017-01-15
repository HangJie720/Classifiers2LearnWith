"""Some tools for displaying information to users."""

from __future__ import division, print_function, absolute_import
import numpy as np


def printmat(arr, row_labels=None, col_labels=None):
    """pretty print a 2d array (optionally) with column and labels"""

    if not row_labels:
        row_labels = ['']*np.array(arr).shape[0]
    if not col_labels:
        col_labels = ['']*np.array(arr).shape[0]

    col_widths = [max(len(str(x)) for x in col) for col in
                                                    np.array(arr).transpose()]
    if col_labels:
        col_widths = [max(w, len(str(l))) for w, l in
                                 zip(col_widths, col_labels)]
    if row_labels:
        rlw = max(len(str(l)) for l in row_labels)
    else:
        rlw = 0

    def pad(x, desired_length, pad=' '):
        if desired_length < len(str(x)):
            raise Exception("desired_length<len(str(x))")
        return pad*(desired_length - len(str(x))) + str(x)

    # Print
    print(pad('', rlw) + '   ' + ' '.join(pad(l, w) for l, w in
                                           zip(col_labels, col_widths)))
    print(pad('', rlw) + '   ' + ' '.join('-'*w for l, w in
                                       zip(col_labels, col_widths)))
    for l, row in zip(row_labels, arr):
        print(pad(l, rlw) + ' | ' + ' '.join(pad(x, w) for x, w in
                                                     zip(row, col_widths)))