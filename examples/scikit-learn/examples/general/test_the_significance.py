#---------------------------------------------------------------#
# Project: Test with permutations the significance of a classification score
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

print(__doc__)

import numpy as np
import pylab as pl

from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn import datasets

def loadData():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    n_classes = np.unique(y).size
    return X, y, n_classes

def addNoise(X):
    random = np.random.RandomState(seed=0)
    E = random.normal(size=(len(X), 2200))
    X = np.c_[X, E]
    return X

def crossValidation(y):
    cv = StratifiedKFold(y, 2)
    return cv

def createSVM():
    svm = SVC(kernel='linear')
    return svm



def computeScore(svm, X, y, cv):
    score, permutation_scores, pvalue = permutation_test_score(svm, \
                                                               X, y, \
                                                               scoring='accuracy', \
                                                               cv=cv, \
                                                               n_permutations=100, \
                                                               n_jobs=1)
    print("Classification score %s (pvalue: %s)" % (score, pvalue))
    return score, permutation_scores, pvalue


def plotHist(score, permutation_scores, pvalue, n_classes):

    pl.hist(permutation_scores, 20, label='Permutation scores')
    ylim = pl.ylim()

    pl.plot(2 * [score], ylim, '--g', \
                               linewidth=3, \
                               label='Classification Score (pvalue %s)' % pvalue)

    pl.plot(2 * [1. / n_classes], ylim, '--k', \
                                        linewidth=3, \
                                        label='Luck')
    
    pl.ylim(ylim)
    pl.legend()
    pl.xlabel('Score')
    pl.show()


def test():
    X, y, n_classes = loadData()
    X = addNoise(X)
    cv = crossValidation(y)
    svm = createSVM()

    score, permutation_scores, pvalue = computeScore(svm, X, y, cv)
    plotHist(score, permutation_scores, pvalue, n_classes)


if __name__ == '__main__':
    test()



