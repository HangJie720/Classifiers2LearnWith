#PROGRESS - this code has an issue I think... doesn't seem to converge like it should
from __future__ import print_function, division
import numpy as np
from scipy.io import loadmat
from numpy.linalg import norm
from warnings import warn
from time import time
from operator import itemgetter
import os

### fixed learning rate chosen by cross-validation
learning_rates = [1]
# learning_rates = [10, 1, 0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-6]
batch_size = 128  # set to 0 to use non-stochastic gradient descent 
max_its = 100000
beta_tol = 1e-2
dloss_tol = 1

rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usps_dict = loadmat(os.path.join(rootdir, 'data', 'usps', 'USPS.mat'))
X = usps_dict['fea']
y = usps_dict['gnd'].ravel()
distinct_labels = set(y)

# Flatten images
m, h, w = X.shape
n = w*h
X = X.reshape((m, n))

# normalize data
# X = X - mean(X);  % hurts accuracy
# X = X./mean(sum((X-mean(X)).^2));  % doesn't help accuracy

m_train = int(0.6*m)
X_train = X[:m_train]
y_train = y[:m_train]

m_valid = int(0.2*m)
X_valid = X[m_train : m_train + m_valid + 1]
y_valid = y[m_train : m_train + m_valid + 1]

m_test = m - m_valid - m_train
X_test = X[m_train + m_valid : len(X) + 1]
y_test = y[m_train + m_valid : len(y) + 1]


def gradient_descent(w0, x, y, df, f, alpha=1e-2, wtol=1e-2, 
                     dftol=1, maxsteps=1000, batchsize=None,
                     backtracking_rate=0.5):
    if not f:
        backtracking_rate = None

    x = np.matrix(x)
    y = np.array(y)
    
    if batchsize:
        indices = range(x.shape[0])
    else:
        xbatch = x
        ybatch = y

    step = 0
    while step < maxsteps:
        # Update
        if step:
            w0 = w1
        step += 1

        # Prepare batch
        if batchsize:
            batch_indices = np.random.choice(indices, batchsize)
            xbatch = x[batch_indices]
            ybatch = y[batch_indices]

        # Step
        df0 = df(w0, xbatch, ybatch)
        w1 = w0 - alpha*df0

        # backtracking
        if backtracking_rate:
            t = backtracking_rate
            f0 = f(w0, xbatch, ybatch)
            f1 = f(w1, xbatch, ybatch)
            alpha0 = alpha
            while 1:
                if f0 - f1 < alpha0*t:
                    break
                alpha0 *= t
                w1 = w0 - alpha0*df0
                f1 = f(w1, xbatch, ybatch)


        # Check if tolerances are satisfied yet
        dw_mag = norm(w1 - w0)
        if dw_mag < wtol and dw_mag/alpha < dftol:
            df1 = df(w1, xbatch, ybatch)
            if norm(df1) < dftol:
                df0 = df(w0, xbatch, ybatch)
                return (w1 if norm(df1) < norm(df0) else w0)

        # Report Progess
        if not step % 100:
            print("\tstep {} : ||w1 - w0||_2 = {} and ||df||_2 = {}"
                  "".format(step, norm(w1 - w0), norm(df1)))

    df0 = df(w0, x, y)
    df1 = df(w1, x, y)
    print("")
    mes = ("Maximum steps reached with "
           "||w1 - w0||_2 = {} and ||df||_2 = {}\n"
           "".format(dw_mag, norm(df1)))
    warn(mes)
    return (w1 if norm(df1) < norm(df0) else w0)





def flatten(list_2_flatten):
    return [item for sublist in list_2_flatten for item in sublist]


def onehot(data, labelset=distinct_labels):
    return [[(1 if l == y else 0) for l in labelset] for y in data]

def getdiagnostic(beta, x, y, nn=lambda z: 1/(1 + np.exp(-z))):
    beta = np.matrix(beta)
    sf = np.array(nn(x*beta.T)).ravel()
    
    if set(y) == set([0,1]):
        try:
            yf = flatten(y)
        except:
            yf = y
    else:
        yf = flatten(onehot(y))
    over50percent = [(1 if p > .5 else 0) for p in sf]
    tp = sum(1 for i in xrange(len(yf)) if 1 == over50percent[i] == yf[i])
    fp = sum(1 for i in xrange(len(yf)) if 1 == over50percent[i] != yf[i])
    tn = sum(1 for i in xrange(len(yf)) if 0 == over50percent[i] == yf[i])
    fn = sum(1 for i in xrange(len(yf)) if 0 == over50percent[i] != yf[i])
    diag = {
        'tp' : tp,
        'fp' : fp,
        'tn' : tn,
        'fn' : fn,
        'ppv' : tp/(tp + fp),  # positive predictive value (precision)
        'tpr' : tp/(tp + fn),  # true positive rate (recall, sensitivity, hit rate)
        'tnr' : tn/(tn + fp),  # true negative rate (specificity)
        'npv' : tn/(tn + fn),  # negative predictive value
        'fpr' : fp/(fp + tn),  # false positive rate (fall-out)
        'fnr' : fn/(fn + tp)  # false negative rate (miss rate)
        }

    # So division by zero response can be controlled with np.seterr()
    for key, val in diag.items():
        diag[key] = np.float32(val)


    diag.update({
        'fdr' : 1 - diag['ppv'],  # false discovery rate
        'accuracy' : (tp + tn)/(tp + tn + fp + fn),
        'f1' : 2 * diag['ppv'] * diag['tpr'] / (diag['ppv'] + diag['tpr']),
        # Mathews Correlation Coefficient
        'mcc' : (tp*tn - fp*fn)/np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)),
        'informedness' : diag['tpr'] + diag['tnr'] - 1,
        'markedness' : diag['ppv'] + diag['npv'] - 1
        })
    diag.update({
 
        })
    return diag


def score(beta, x, y, full=True, nn=lambda z: 1/(1 + np.exp(-z))):
    assert len(x) == len(y)
    probs, y_pred = predict(x, beta)
    ct = np.count_nonzero(y == y_pred) # number correct
    print("Accuracy: {} / {} = {}".format(ct, len(y), ct/len(y)))
    if full:
        diagnostic = getdiagnostic(beta, x, y, nn=nn)
        # estimate of the chance that a predicted positive is positive
        print("Precision:", diagnostic['ppv']) 
        # estimate of the chance that a positive will be predicted as positive
        print("Recall:", diagnostic['tpr'])
        print("F1 Score:", diagnostic['f1'])
        print("MCC:", diagnostic['mcc'])
    return ct


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def predict(x, beta, nn=sigmoid):
    beta = np.matrix(beta)
    s = nn(x*beta.T)
    probs, predicted_labels = zip(*[(np.max(ps), np.argmax(ps)) for ps in s])
    return probs, predicted_labels


def loss(beta, x, y, nn=sigmoid):
    probs, y_pred = predict(x, beta) 
    d = np.array(y-probs)
    return sum(d * d)/len(y)


def dloss(beta, x, y):
    beta = np.matrix(beta)
    s = np.array(1/(1 + np.exp(-x*beta.T))).reshape((len(y),))
    return np.matrix((y-s) * s * (1-s)) * x


# Train
tr_time = time()
print("Training using {} in batches of {} examples..."
      "".format(len(X_train), batch_size))
beta_init = np.array([1]*len(X_train[0]))
beta = []
totloss = 0
for c in distinct_labels:
    
    y_train_c = (y_train == c) * 1
    y_valid_c = (y_valid == c) * 1
    beta_results = []
    for alpha_idx, alpha in enumerate(learning_rates):
        
        print("({}/{}) Training using {} examples and alpha = {}..."
              "".format(alpha_idx + 1, 
                    len(learning_rates), 
                    len(X_train), alpha))
        
        beta_res = gradient_descent(beta_init, X_train, y_train_c, dloss, 
                                    alpha=alpha, wtol=beta_tol, dftol=dloss_tol, 
                                    maxsteps=max_its, batchsize=batch_size,
                                    f=loss, backtracking_rate=0.5)
        
        tl = loss(beta_res, X_valid, y_valid_c)
        beta_results.append((beta_res, tl, alpha))

        print("\tFinished training label {} with alpha = {}\n"
              " : Total validation loss = {}\n"
              " : Validation "
              "".format(c, alpha, tl), end="")
        score(beta_res, X_valid, y_valid_c, full=False)
    
    beta_c, totloss_c, alpha_c = min(beta_results, key=itemgetter(1))
    totloss += totloss_c
    input()
    beta.append(np.array(beta_c).reshape(beta_c.shape[1]))

    print("\tFinished training label {} : chose alpha = {}"
      "".format(c, alpha_c))
beta = np.array(beta)
print("Finished training in {} seconds with training total loss = {}.\n"
      "".format(time() - tr_time, totloss))

# Test
print("Test ", end="")
score(beta, X_test, y_test)