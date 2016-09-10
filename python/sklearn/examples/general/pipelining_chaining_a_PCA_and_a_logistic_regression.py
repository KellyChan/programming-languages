#---------------------------------------------------------------#
# Project: Pipelining: chaining a PCA and a logistic regression
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

print(__doc__)

import numpy as np
import pylab as pl

from sklearn import linear_model, decomposition, datasets
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

def createLogistic():
    logistic = linear_model.LogisticRegression()
    return logistic

def createPCA():
    pca = decomposition.PCA()
    return pca

def createPipeline(logistic, pca):
    pipe = Pipeline(steps=[('pca', pca), \
                           ('logistic', logistic)])
    return pipe

def loadData():
    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target
    return X_digits, y_digits


def predict(pipe, X_digits, y_digits):

    n_components = [20, 40, 64]
    Cs = np.logspace(-4, 4, 3)

    estimator = GridSearchCV(pipe, \
                             dict(pca__n_components=n_components, \
                                  logistic__C=Cs))
    
    y_pred = estimator.fit(X_digits, y_digits)
    return estimator, y_pred


def plotPCA(estimator, pca, X_digits):

    pca.fit(X_digits)

    pl.figure(1, figsize=(4, 3))
    pl.clf()
    pl.axes([.2, .2, .7, .7])
    pl.plot(pca.explained_variance_, linewidth=2)
    pl.axis('tight')
    pl.xlabel('n_components')
    pl.ylabel('explained_variance_')

    pl.axvline(estimator.best_estimator_.named_steps['pca'].n_components, \
               linestyle=':', \
               label='n_components chosen')
    pl.legend(prop=dict(size=12))
    pl.show()



def test():

    logistic = createLogistic()
    pca = createPCA()
    pipe = createPipeline(logistic, pca)

    X_digits, y_digits = loadData()

    estimator, y_pred = predict(pipe, X_digits, y_digits)
    plotPCA(estimator, pca, X_digits)

if __name__ == '__main__':
    test()



