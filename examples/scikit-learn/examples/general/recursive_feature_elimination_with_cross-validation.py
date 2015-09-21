#-----------------------------------------------------------#
# Project: Recursive Feature Elimination
# Author: Kelly Chan
# Date: Apr 22 2014
#-----------------------------------------------------------#

print(__doc__)

from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.metrics import zero_one_loss

import pylab as pl

def loadData():
    # build a classification task using 3 informative features
    X, y = make_classification(n_samples=1000, \
                               n_features=25, \
                               n_informative=3, \
                               n_redundant=2, \
                               n_repeated=0, \
                               n_classes=8, \
                               n_clusters_per_class=1, \
                               random_state=0)
    return X, y



def createRFECV(y):
    # create the RFE objects and compute a cross-validated score
    svc = SVC(kernel='linear')
    rfecv = RFECV(estimator=svc, \
                  step=1, \
                  cv=StratifiedKFold(y, 2), \
                  scoring='accuracy')
    return rfecv

def predict(rfecv, X, y):
    return rfecv.fit(X, y)


def plotRFECV(rfecv):
    # plotting number of features VS cross-validation scores
    pl.figure()
    pl.xlabel("Number of features selected")
    pl.ylabel("Cross validation score (nb of misclassifications)")
    pl.plot(range(1, len(rfecv.grid_scores_) + 1), \
            rfecv.grid_scores_)
    pl.show()

def test():
    X, y = loadData()
    rfecv = createRFECV(y)
    #print("Optimal number of features: %d" % rfecv.n_features_)
    
    target = predict(rfecv, X, y)
    plotRFECV(rfecv)



if __name__ == '__main__':
    test()
