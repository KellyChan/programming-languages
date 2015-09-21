#-----------------------------------------------------------#
# Project: Precision Recall
# Author: Kelly Chan
# Date: Apr 22 2014
#-----------------------------------------------------------#
 

import random
import pylab as pl
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


def loadData():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X, y = X[y != 2], y[y != 2]  # keep also 2 classes (0/1)
    return X, y

def sampling(X, y):

    n_samples, n_features = X.shape
    half = int(n_samples / 2)

    p = range(n_samples)  # shuffle samples
    random.seed(0)
    random.shuffle(p)
    X, y = X[p], y[p]


    # add noise
    np.random.seed(0)
    X = np.c_[X, np.random.randn(n_samples, 200 * n_features)]

    return X, y, half



def createSVM():
    classifier = svm.SVC(kernel='linear', \
                         probability=True, \
                         random_state=0)
    return classifier

def classify(classifier, X, y, half):
    clf = classifier.fit(X[:half], y[:half])
    probas_ = clf.predict_proba(X[half:])
    return probas_

def computeAUC(y, probas_, half):
    precision, recall, thresholds = precision_recall_curve(y[half:], \
                                                           probas_[:, 1])
    area = auc(recall, precision)
    print("Area Under Curve: %.2f" % area)
    return precision, recall, area

def plotAUC(precision, recall, area):
    pl.clf()
    pl.plot(recall, precision, label='Precision-Recall curve')
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title('Precision-Recall example: AUC=%.2f' % area)
    pl.legend(loc='lower left')
    pl.show()


def test():
    X, y = loadData()
    X, y, half = sampling(X, y)

    classifier = createSVM()
    probas_ = classify(classifier, X, y, half)
    
    precision, recall, area = computeAUC(y, probas_, half)
    plotAUC(precision, recall, area)



if __name__ == '__main__':
    test()
