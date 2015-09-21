#---------------------------------------------------------------#
# Project: Receiver operating characteristic (ROC) with cross validation
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

print(__doc__)

import numpy as np
import pylab as pl
from scipy import interp

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

def loadData():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape
    
    return X, y, n_samples, n_features

def addNoise(X, n_samples, n_features):
    X = np.c_[X, np.random.randn(n_samples, 200 * n_features)]
    return X

def crossValidation(y):
    cv = StratifiedKFold(y, n_folds=6)
    return cv

def createSVM():
    classifier = svm.SVC(kernel='linear', probability=True, random_state=0)
    return classifier

def plotROC(classifier, cv, X, y):

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])

        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

        roc_auc = auc(fpr, tpr)
        pl.plot(fpr, tpr, lw=1, \
                          label='ROC fold %d (area = %.2f)' % (i, roc_auc))

    pl.plot([0, 1], [0, 1], '--', \
                            color=(0.6, 0.6, 0.6), \
                            label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    pl.plot(mean_fpr, mean_tpr, 'k--', \
                                label='Mean ROC (area = %.2f)' % mean_auc, lw=2)

    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc='lower right')
    pl.show()


def test():
    X, y, n_samples, n_features = loadData()
    X = addNoise(X, n_samples, n_features)
    cv = crossValidation(y)
    
    classifier = createSVM()
    plotROC(classifier, cv, X, y)

if __name__ == '__main__':
    test()

