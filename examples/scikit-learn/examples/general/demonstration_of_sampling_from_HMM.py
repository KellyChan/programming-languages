#---------------------------------------------------------------#
# Project: Demonstration of sampling from HMM
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import hmm


def loadData():
    
    # population probability
    start_prob = np.array([0.6, 0.3, 0.1, 0.0])
    
    # transition matrix
    trans_mat = np.array([[0.7, 0.2, 0.0, 0.1],
                          [0.3, 0.5, 0.2, 0.0],
                          [0.0, 0.3, 0.5, 0.2],
                          [0.2, 0.0, 0.2, 0.6]])
    
    # mean of each component
    means = np.array([[0.0,  0.0],
                      [0.0, 11.0],
                      [9.0, 10.0],
                      [11.0, -1.0],
                      ])

    # covariance of each component
    covars = .5 * np.tile(np.identity(2), (4, 1, 1))

    return start_prob, trans_mat, means, covars


def createHMM(start_prob, trans_mat, means, covars):
    model = hmm.GaussianHMM(4, 'full', start_prob, trans_mat, random_state=42)
    model.means_ = means
    model.covars_ = covars
    return model, model.means_, model.covars_

def sampling(model):
    X, Z = model.sample(500)
    return X, Z

def plotData(X, means):
    plt.plot(X[:, 0], X[:, 1], "-o", \
                               label="observations", \
                               ms=6, \
                               mfc="orange", \
                               alpha=0.7)
    
    for i, m in enumerate(means):
        plt.text(m[0], m[1], "Component %i" % (i + 1), \
                             size=17, \
                             horizontalalignment='center', \
                             bbox=dict(alpha=.7, facecolor='w'))

    plt.legend(loc='best')
    plt.show()


def test():
    start_prob, trans_mat, means, covars = loadData()
    model, model.means_, model.covars_ = createHMM(start_prob, trans_mat, means, covars)
    
    X, Z = sampling(model)
    plotData(X, model.means_)

if __name__ == '__main__':
    test()



