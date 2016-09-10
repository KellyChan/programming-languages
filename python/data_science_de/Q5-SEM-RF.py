"""
Project: Data Science
Subject: Machine Learning - SEM

Author: Kelly Chan
Date: May 10 2014
"""

tabPath = "path/outputs/sem/tables/"

import numpy as np
import pylab as pl
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

def loadData(datafile):
    return pd.read_csv(datafile)

def splitData(data, rate):

    index = int(len(data) * rate)

    train = data.iloc[:index]
    test = data.iloc[index:]
    return train, test

def createRFRegression():
    rf = RandomForestRegressor(n_estimators=20)
    return rf

def predictRF(rf, train, test, cols, target): 
    rf.fit(train[cols], train[target])
    return rf.predict(test[cols])

def evaluateRF(rf, test, cols, target):
    r2 = r2_score(test[target], rf.predict(test[cols]))
    mse = np.mean((test[target] - rf.predict(test[cols]))**2)
    return r2, mse

def plotRF(rf, test, cols, target, r2):

    pl.scatter(test[target], rf.predict(test[cols]))
    pl.plot(np.arange(8, 15), np.arange(8, 15), label="r^2=" + str(r2), c="r")
    pl.legend(loc="lower right")
    pl.title("RandomForest Regression: %s" % target)
    pl.show()

def predict(train, test, target, cols):

    rf = createRFRegression()
    yhat = predictRF(rf, train, test, cols, target)
    r2, mse = evaluateRF(rf, test, cols, target)

    plotRF(rf, test, cols, target, r2)

    return yhat, r2, mse


def main():

    data = loadData(tabPath + "clean_data.csv")
    train, test = splitData(data, 0.8)




    target = 'Order Value'
    cols = ['Campaign Clicks', \
            'Visitors', 'New Visitors', \
            'Bounces', 'Entry Rate %', 'Exit Rate %', \
            'OP Visits w Catalogue %', \
            'OP Visits w Search %', \
            'OP Visits w Product %', \
            'OP Visits w step Cart %', \
            'Campaign Lifecycle Contacts']
    #train = data
    #test = data
    yhat, r2, mse = predict(train, test, target, cols)

    print yhat, r2, mse



if __name__ == '__main__':
    main()
