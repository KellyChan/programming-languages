#---------------------------------------------------------------#
# Project: Train error vs Test error
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

print(__doc__)

import numpy as np
import pylab as pl
from sklearn import linear_model


def loadData():
    
    n_samples_train = 75
    n_samples_test = 150
    n_features = 500

    np.random.seed(0)
    coef = np.random.randn(n_features)
    coef[50:] = 0.0  # only the top 10 features are impacting the model

    X = np.random.randn(n_samples_train + n_samples_test, n_features)
    y = np.dot(X, coef)

    return X, y, n_samples_train, n_samples_test, coef

def splitData(X, y, n_samples_train):
    X_train, X_test = X[:n_samples_train], X[n_samples_train:]
    y_train, y_test = y[:n_samples_train], y[n_samples_train:]
    return X_train, X_test, y_train, y_test


def computeError(X, y, X_train, X_test, y_train, y_test):
    
    alphas = np.logspace(-5, 1, 60)
    enet = linear_model.ElasticNet(l1_ratio=0.7)
    
    train_errors = list()
    test_errors = list()

    for alpha in alphas:
        enet.set_params(alpha=alpha)
        enet.fit(X_train, y_train)
        train_errors.append(enet.score(X_train, y_train))
        test_errors.append(enet.score(X_test, y_test))

    i_alpha_optim = np.argmax(test_errors)
    alpha_optim = alphas[i_alpha_optim]
    print("Optimal regularization parameter: %s" % alpha_optim)

    # estimate the coef_ on full data with optimal regularization parameter 
    enet.set_params(alpha=alpha_optim)
    coef_ = enet.fit(X, y).coef_

    return train_errors, test_errors, alphas, alpha_optim, coef_



def plotErrors(train_errors, test_errors, alphas, alpha_optim):
    
    pl.subplot(2, 1, 1)
    
    pl.semilogx(alphas, train_errors, label='Train')
    pl.semilogx(alphas, test_errors, label='Test')
    
    pl.vlines(alpha_optim, \
              pl.ylim()[0], \
              np.max(test_errors), \
              color='k', \
              linewidth=3, \
              label='Optimum on test')
    
    pl.legend(loc='lower left')
    pl.ylim([0, 1.2])
    pl.xlabel('Regularization parameter')
    pl.ylabel('Performance')


def plotCoef(coef, coef_):
    
    pl.subplot(2, 1, 2)
    
    pl.plot(coef, label='True coef')
    pl.plot(coef_, label='Estimated coef')
    
    pl.legend()
    pl.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.26)
    pl.show()


def test():
    X, y, n_samples_train, n_samples_test, coef = loadData()
    X_train, X_test, y_train, y_test = splitData(X, y, n_samples_train)

    train_errors, test_errors, alphas, alpha_optim, coef_ = computeError(X, y, X_train, X_test, y_train, y_test)
    
    plotErrors(train_errors, test_errors, alphas, alpha_optim)
    plotCoef(coef, coef_)

if __name__ == '__main__':
    test()


