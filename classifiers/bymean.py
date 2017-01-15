from operator import itemgetter
import numpy as np
norm2 = np.linalg.norm


class DistanceFromMeanClassifier(object):
    """Classify by distance from mean, simple as that."""
    def __init__(self):
        self.distinct_labels = None
        self.means = None  # mean vector of class
        self.sds = None   # sd of distances in class from class mean

    def train(self, X, Y):
        self.distinct_labels = np.unique(Y)
        self.means = np.empty((self.distinct_labels.shape[0], X.shape[1]))
        self.sds = np.empty(self.distinct_labels.shape)
        for l in self.distinct_labels:
            Xl = X[np.where(Y == l)]  # examples labeled l
            mean_l = np.mean(Xl, axis=0)
            s_l = np.std([norm2(x - mean_l) for x in Xl])
            self.sds[l] = s_l
            self.means[l] = mean_l

        # Check how far away means are from each other.
        for i, mi in enumerate(self.means):
            d2i = [(j, norm2(mi - mj)) for j, mj in enumerate(self.means) if
                   i != j]
            j, dij = min(d2i, key=itemgetter(1))
            si, sj = self.sds[i], self.sds[j]
            print("label {} :: nearest label = {} :: dist = {}) :: "
                  "sd_{} = {} :: sd_{} = {}"
                  "".format(i, j, dij, i, si, j, sj))

    def predict(self, X):
        pred_indices = np.argmin([norm2(X - m, axis=1) for m in self.means],
                                 axis=0)
        return self.distinct_labels[pred_indices]
