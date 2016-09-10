#---------------------------------------------------------------#
# Project: Univariate Feature Selection
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

print(__doc__)

import numpy as np
import pylab as pl

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif

def loadData():
    iris = datasets.load_iris()
    E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))
    X = np.hstack((iris.data, E))
    y = iris.target
    return X, y

def selectFeatures(X, y):
    # feature selection with F-test for feature scoring
    # 10% most significant features
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(X, y)

    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()

    return selector, scores

def createSVM():
    clf = svm.SVC(kernel='linear')
    return clf

def predictSVM(clf, X, y):

    yHat = clf.fit(X, y)  

    svm_weights = (clf.coef_ ** 2).sum(axis=0)
    svm_weights /= svm_weights.max()

    return yHat, svm_weights


def plotScores(X, scores, selector, svm_weights, svm_weights_selected):

    X_indices = np.arange(X.shape[-1])

    pl.figure(1)
    pl.clf()

    pl.bar(X_indices - .45, scores, width=.2, \
                                    label=r'Univariate score ($-Log(p_{value})$)', \
                                    color='g')

    pl.bar(X_indices - .25, svm_weights, width=.2, \
                                         label='SVM weight', \
                                         color='r')

    pl.bar(X_indices[selector.get_support()] - .05, \
           svm_weights_selected, width=.2, \
                                 label='SVM weights after selection', \
                                 color='b')

    pl.title("Comparing feature selection")
    pl.xlabel('Feature number')
    pl.yticks(())
    pl.axis('tight')
    pl.legend(loc='upper right')
    pl.show()


def test():

    X, y = loadData()
    selector, scores = selectFeatures(X, y)
    
    clf = createSVM()
    yHat, svm_weights = predictSVM(clf, X, y)
    yHat_selected, svm_weights_selected = predictSVM(clf, selector.transform(X), y)

    plotScores(X, scores, selector, svm_weights, svm_weights_selected)


if __name__ == '__main__':
    test()



