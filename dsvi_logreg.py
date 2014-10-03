
import os
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randn, rand
from sklearn.linear_model import LogisticRegression


def logistic_sigmoid(x):
    return 1. / (1. + np.exp(-x))


def log_logistic_sigmoid(x):
    out = np.empty_like(x)
    out[x>0.] = -np.log(1. + np.exp(-x[x>0.]))
    out[x<=0.] = x[x<=0.] - np.log(1. + np.exp(x[x<=0.]))
    return out


def dsvi_logreg_full(Y, X, mu=None, C=None, niter=100, kappa=1., tau=0.,
                     m0=None, S0=None):
    """ DSVI for Bayesian logistic regression.

        No minibatches for now.
        
        Y : N-vector of binary responses coded as {-1, 1}
        X : N x p design matrix
        mu : p-vector, initial variational mean
        C  : p x p lower Cholesky factor of variational posterior covariance
        niter : number of iterations
        kappa0 : float, learning rate exponent
        tau0 : float, learning rate offset
        m0 : prior mean of parameters
        S0 : prior covariance of parameters

        Returns:
          - mu : mean vector of variational approximation
          - C  : Cholesky factor (lower triangle) of covariance of variational
                 approximation
          - ELBO : values of ELBO
    """

    # Setup
    N, p = X.shape
    if mu is None:
        mu = np.zeros(p)
    if C is None:
        C = np.eye(p)
    if m0 is None:
        m0 = np.zeros(p)
    if S0 is None:
        S0 = np.eye(p)

    L0 = np.linalg.inv(S0)
    L0m0 = np.dot(L0, m0)

    iELBO = np.empty(niter)

    # Just constant for now
    #rho = 1e-3
    rho = (1. + tau) ** (-kappa)

    for it in xrange(niter):
        
        # Try sampling multiple points again
        z = randn(p)
        th = np.dot(C, z) + mu

        # Specialized for logistic
        w = -Y * np.dot(X, th)
        YX = Y[:,np.newaxis]*X
        S = logistic_sigmoid(w)
        grad_logreg = np.sum(S[:,np.newaxis] * YX, axis=0) \
                       - np.dot(L0, th) + L0m0

        mu += rho * grad_logreg
        dC = np.outer(grad_logreg, z)
        dC = np.tril(dC) + np.diag(1./np.diag(C))
        C += 0.1*rho * dC
        C = np.tril(C)
        keep = np.diag(C).copy()
        keep[keep<=1e-4] = 1e-4
        C = C + (np.diag(keep - np.diag(C)))

        rho = (1. + it + tau) ** (-kappa)

        # Evaluate instantaneous ELBO (don't need entropy of \phi(z) b/c
        # constant w.r.t. variational params).
        iELBO[it] = np.sum(log_logistic_sigmoid(-w), axis=0) \
                    - 0.5*np.dot(th, np.dot(L0, th)) \
                    + np.dot(m0, np.dot(L0, th))
        iELBO[it] += np.sum(np.log(np.diag(C)))
        # Entropy
        iELBO[it] += 0.5*p + 0.5*p*np.log(2.*np.pi)

    return mu, C, iELBO


def dsvi_pred_mc(X, mu, C):
    """ Predictive classification probability using Monte Carlo sample fro q(w)
        
        X : N_test x p, design matrix for test set
        mu : p-vector, mean of q(w)
        C : p x p, lower Cholesky factor of covariance of q(w)
    """
    pass


if __name__ == "__main__":

    np.random.seed(8675309)

    # sklearn synthetic data
    #from sklearn.datasets import make_classification
    #dargs = {'n_samples': 100, 'n_features': 2, 'n_informative': 2,
    #         'n_redundant': 0, 'n_clusters_per_class': 1}
    #p = 2
    #X, Y = make_classification(**dargs)
    # Recode as {-1, 1}
    #Y = 2*Y - 1  # Label as {-1,1}

    # Simple synthetic data
    #N = 100
    #p = 2
    #X = np.empty((N, 2))
    #X[:,0] = 1.
    #X[:,1] = np.random.randn(N)
    #th0 = np.array([1., 1.])
    #Y = np.squeeze(np.round(logistic_sigmoid(np.dot(X, th0))))
    # Recode as {-1, 1}
    #Y = 2*Y - 1  # Label as {-1,1}

    # Synthetic example from Drugowitsch
    #p = 3
    #N = 100
    #N_test = 1000
    #X_scale = 5
    #th = randn(p)
    #X = np.empty((N, p))
    #X[:,0] = 1.
    #X[:,1] = X_scale * (rand(N) - 0.5)
    #X[:,2] = X_scale * (rand(N) - 0.5) - (th[0] + X[:,1]*th[1]) / th[2]

    #X_test = np.empty((N_test, p))
    #X_test[:,0] = 1.
    #X_test[:,1] = X_scale * (rand(N_test) - 0.5)
    #X_test[:,2] = X_scale * (rand(N_test) - 0.5) - (th[0] + X_test[:,1]*th[1]) / th[2]

    #p_y = logistic_sigmoid(np.dot(X, th))
    #Y = 2 * (rand(N) < p_y) - 1
    #Y_test = 2 * (rand(N_test) < logistic_sigmoid(np.dot(X_test, th))) - 1

    # Pima indians
    import pandas as pd
    from sklearn.cross_validation import train_test_split
    dpath = "~/work/data/uci_pima_indians/pima-indians-diabetes-filtered.csv"
    df = pd.read_csv(os.path.expanduser(dpath))
    X_full = df.ix[:,df.columns != 'CLASS'].values
    N, p = X_full.shape
    tmp = np.empty((N, p+1))
    tmp[:,0] = 1.
    tmp[:,1:] = X_full
    X_full = tmp
    Y_full = df['CLASS'].values
    Y_full = 2*Y_full - 1

    X_full -= np.mean(X_full, axis=0)
    X_full[:,1:] /= np.std(X_full[:,1:], axis=0)

    X, X_test, Y, Y_test = train_test_split(X_full, Y_full, test_size=0.2,
                                            random_state=8675309)
    N, p = X_full.shape
    N_test = X_test.shape[0]

    niter = 5000
    wlen = 200
    mu, C, F = dsvi_logreg_full(Y, X, mu=None, C=np.eye(p), niter=niter,
                                kappa=.7, tau=50)

    Fsm = np.zeros(niter-wlen)
    for i in xrange(Fsm.shape[0]):
        Fsm[i] = np.mean(F[i:i+wlen])

    # sklearn logistic regression
    lr = LogisticRegression(penalty='l2', fit_intercept=False, C=0.5)
    lr.fit(X, Y)

    plt.plot(Fsm)
    plt.title('Smoothed instantaneous ELBO')

    # Plot true parameters -- only for synthetic
    #plt.figure()
    #plt.plot(th, mu, '.')
    #plt.hold(True)
    #plt.plot(th, np.squeeze(lr.coef_), '.')
    #plt.plot(th, th, '.')
    #plt.show()

    dr_tr = (np.round(logistic_sigmoid(np.dot(X, mu))) - 0.5)*2.
    #opr_tr = (np.round(logistic_sigmoid(np.dot(X, np.squeeze(lr.coef_)))) - 0.5)*2.
    opr_tr = lr.predict(X)
    dr_te = (np.round(logistic_sigmoid(np.dot(X_test, mu))) - 0.5)*2.
    #opr_te = (np.round(logistic_sigmoid(np.dot(X_test, np.squeeze(lr.coef_)))) - 0.5)*2.
    opr_te = lr.predict(X_test)

    print "Train Accuracy:"
    print "  dsvi: %.2f" % (1. - np.sum(np.abs(Y - dr_tr) > 0.) / float(N),)
    print "  LR: %.2f" % (1. - np.sum(np.abs(Y - opr_tr) > 0.) / float(N),)

    print "Test Accuracy:"
    print "  dsvi: %.2f" % (1. - np.sum(np.abs(Y_test - dr_te) > 0.) / float(N_test),)
    print "  LR: %.2f" % (1. - np.sum(np.abs(Y_test - opr_te) > 0.) / float(N_test),)
