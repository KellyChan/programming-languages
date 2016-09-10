#-----------------------------------------------------------#
# Project: Recursive Feature Elimination
# Author: Kelly Chan
# Date: Apr 22 2014
#-----------------------------------------------------------#

print(__doc__)

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE

import pylab as pl

def loadData():

    # load the digits dataset
    digits = load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    y = digits.target
    
    return digits, X, y

def createRFE():
    svc = SVC(kernel='linear', C=1)
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    return rfe

def rankPixels(rfe, digits, X, y):
    rfe.fit(X, y)
    ranking = rfe.ranking_.reshape(digits.images[0].shape)
    return ranking

def plotRankPixels(ranking):
    pl.matshow(ranking)
    pl.colorbar()
    pl.title("Ranking of pixels with RFE")
    pl.show()


def test():
    digits, X, y = loadData()
    rfe = createRFE()
    ranking = rankPixels(rfe, digits, X, y)
    plotRankPixels(ranking)

if __name__ == '__main__':
    test()
