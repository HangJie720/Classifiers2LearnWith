"""Some diagnostic tools."""

from __future__ import division, print_function, absolute_import
from collections import OrderedDict
import numpy as np


from .display_tools import printmat


def getbinarydiagnostic(labels, predictions):
    """returns a dictionary of various ratios used for diagnostic purposes."""

    tp = sum(y == pred for y, pred in zip(labels, predictions) if pred)
    fp = sum(y != pred for y, pred in zip(labels, predictions) if pred)
    tn = sum(y == pred for y, pred in zip(labels, predictions) if not pred)
    fn = sum(y != pred for y, pred in zip(labels, predictions) if not pred)
    tp, fp, tn, fn = [np.float32(x) for x in (tp, fp, tn, fn)]
    old_seterr = np.seterr(divide="ignore")
    diag = OrderedDict()
    diag.update({
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'ppv': tp / (tp + fp),  # positive predictive value (precision)
        'tpr': tp / (tp + fn),  # true positive rate (recall, sensitivity)
        'tnr': tn / (tn + fp),  # true negative rate (specificity)
        'npv': tn / (tn + fn),  # negative predictive value
        'fpr': fp / (fp + tn),  # false positive rate
        'fnr': fn / (fn + tp)  # false negative rate
    })
    diag.update({
        'recall': diag['tpr'],
        'precision': diag['ppv']
    })
    diag.update({
        'fdr': 1 - diag['ppv'],  # false discovery rate
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'f1': 2 * diag['ppv'] * diag['tpr'] / (diag['ppv'] + diag['tpr']),
        # Mathews Correlation Coefficient
        'mcc': (tp * tn - fp * fn) / np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
        'informedness': diag['tpr'] + diag['tnr'] - 1,
        'markedness': diag['ppv'] + diag['npv'] - 1
    })
    np.seterr(**old_seterr)
    return diag


def getdiagnostics(labels, predictions):
    """returns a dictionary of dictionaries (one for distinct label, unless
    binary) of various ratios used for diagnostic purposes."""
    diags = dict()
    distinct_labels = set(labels)

    # In case of binary labels
    if len(distinct_labels) == 2:
        l0, l1 = distinct_labels
        if l0 and not l1:
            diags[l0] = getbinarydiagnostic(labels, predictions)
            return diags
        if l1 and not l0:
            diags[l1] = getbinarydiagnostic(labels, predictions)
            return diags

    # In case of more than two distinct labels
    for l in set(labels):
        ground_truth = [l == g for g in labels]
        bin_predictions = [l == p for p in predictions]
        diags[l] = getbinarydiagnostic(ground_truth, bin_predictions)
    return diags


class diagnostic(object):
    def __init__(self, labels):
        self.labels = labels
        self.distinct_labels = set(labels)
        self.diagnoses = dict()  # dictionary of dictionaries of dictionaries

    def diagnose(self, predictions, method=None):
        if method is None:
            method = "UnNamed: " + str(len(self.diagnoses))
        self.diagnoses[method] = getdiagnostics(self.labels, predictions)

    def report(self, keys=['recall', 'precision', 'accuracy', 'f1']):
        methods = self.diagnoses.keys()
        # ratios = self.diagnoses.values()[0].values()[0].keys()
        for l in self.distinct_labels:
            printmat([[d[l][k] for k in keys] for d in self.diagnoses.values()],
                     col_labels=keys,
                     row_labels=methods)

            # Find winner
            winners = []
            for measure in keys:
                contest_results = []
                for m in methods:
                    res = self.diagnoses[m][l][measure]
                    if res is not np.NaN:
                        contest_results.append((res, m))
                w = max(contest_results)
                winners.append(['Winner by ' + measure + ':',
                                str(w[1]),
                                str(w[0])])
            printmat(winners)
