#---------------------------------------------------------------#
# Project: Plot classification probability
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

print(__doc__)

import numpy as np
import pylab as pl

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets

def loadData():
    iris = datasets.load_iris()
    X = iris.data[:, 0:2]  # only take the first two features for viz
    y = iris.target
    n_features = X.shape[1]
    return X, y, n_features

def createClassifiers():

    C = 1.0
    classifiers = {'L1 logistic': LogisticRegression(C=C, penalty='l1'), \
                   'L2 logistic': LogisticRegression(C=C, penalty='l2'), \
                   'Linear SVC': SVC(kernel='linear', C=C, probability=True, random_state=0) \
                     }    
    
    n_classifiers = len(classifiers)

    return classifiers, n_classifiers


def plotProb(n_classifiers, classifiers, X, y):

    pl.figure(figsize=(3 * 2, n_classifiers * 2))
    pl.subplots_adjust(bottom=.2, top=.95)

    for index, (name, classifier) in enumerate(classifiers.iteritems()):
        
        classifier.fit(X, y)
        y_pred = classifier.predict(X)

        classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
        print("Classif_rate for %s: %f" % (name, classif_rate))

        # view probabilities
        xx = np.linspace(3, 9, 100)
        yy = np.linspace(1, 5, 100).T
        xx, yy = np.meshgrid(xx, yy)
        Xfull = np.c_[xx.ravel(), yy.ravel()]

        probas = classifier.predict_proba(Xfull)
        n_classes = np.unique(y_pred).size

        for k in range(n_classes):
            pl.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
            pl.title("Class %d" % k )

            if k == 0:
                pl.ylabel(name)

            imshow_handle = pl.imshow(probas[:, k].reshape((100, 100)), \
                                      extent=(3, 9, 1, 5), \
                                      origin='lower')

            pl.xticks(())
            pl.yticks(())
            
            idx = (y_pred == k)
            if idx.any():
                pl.scatter(X[idx, 0], X[idx, 1], marker='o', c='k')


    ax = pl.axes([.15, .04, .7, .05])
    pl.title("Probability")
    pl.colorbar(imshow_handle, cax=ax, orientation='horizontal')
    pl.show()


def test():
    X, y, n_features = loadData()

    classifiers, n_classifiers = createClassifiers()   
    plotProb(n_classifiers, classifiers, X, y)

if __name__ == '__main__':
    test()


