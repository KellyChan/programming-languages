#---------------------------------------------------------------#
# Project: Receiver operating characteristic (ROC)
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

print(__doc__)

import numpy as np
import pylab as pl

from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc


random_state = np.random.RandomState(0)

def loadData():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # make it a binary classification problem by removing the third class
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape

    return X, y, n_samples, n_features


def addNoise(X, n_samples, n_features):
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    return X

def splitData(X, y, n_samples):
    X, y = shuffle(X, y, random_state=random_state)
    half = int(n_samples / 2)
    
    X_train, X_test = X[:half], X[half:]
    y_train, y_test = y[:half], y[half:]
    
    return X_train, X_test, y_train, y_test

def createSVM():
    classifier = svm.SVC(kernel='linear', probability=True, random_state=0)
    return classifier

def classify(classifier, X_train, y_train, X_test):
    probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
    return probas_


def computeROC(y_test, probas_):
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve: %f" % roc_auc)
    return fpr, tpr, roc_auc


def plotROC(fpr, tpr, roc_auc):
    pl.clf()
    pl.plot(fpr, tpr, label="ROC curve (area = %.2f)" % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc='lower right')
    pl.show()


def test():
    X, y, n_samples, n_features = loadData()
    X = addNoise(X, n_samples, n_features)
    X_train, X_test, y_train, y_test = splitData(X, y, n_samples)

    classifier = createSVM()
    probas_ = classify(classifier, X_train, y_train, X_test)

    fpr, tpr, roc_auc = computeROC(y_test, probas_)
    plotROC(fpr, tpr, roc_auc)


if __name__ == '__main__':
    test()
