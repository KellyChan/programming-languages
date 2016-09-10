#---------------------------------------------------------------#
# Project: Imputing missing values before building an estimator
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

import numpy as np

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score


def loadData():
    dataset = load_boston()
    X_full, y_full = dataset.data, dataset.target
    n_samples = X_full.shape[0]
    n_features = X_full.shape[1]
    return X_full, y_full, n_samples, n_features



def addMissing(n_samples, n_features):
    # add missing values in 75% of the lines
    missing_rate = 0.75
    n_missing_samples = np.floor(n_samples * missing_rate)
    missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples, \
                                          dtype=np.bool), \
                                 np.ones(n_missing_samples, \
                                         dtype=np.bool)))

    rng = np.random.RandomState(0)
    rng.shuffle(missing_samples)
    missing_features = rng.randint(0, n_features, n_missing_samples)

    return missing_samples, missing_features

def estimateData(X, y):
    # estimate the score on the entire dataset, with no missing values
    estimator = RandomForestRegressor(random_state=0, n_estimators=100)
    score = cross_val_score(estimator, X, y).mean()
    print("Score with the dataset = %.2f" % score)
    return score

def estimateMissing(X_missing, y_missing):
    estimator = Pipeline([('imputer', Imputer(missing_values=0, \
                                              strategy='mean', \
                                              axis=0)), \
                          ('forest', RandomForestRegressor(random_sate=0, \
                                                           n_estimators=100))])
    score = cross_val_score(estimator, X_missing, y_missing).mean()
    print("Score after imputation of the missing values = %.2f" % score)
    return score


def test():

    X_full, y_full, n_samples, n_features = loadData()
    score_full = estimateData(X_full, y_full)

    missing_samples, missing_features = addMissing(n_samples, n_features)
    # data without the lines containing missing values
    X_filtered = X_full[~missing_samples, :]
    y_filtered = y_full[~missing_samples, :]
    score_filtered = estimateData(X_filtered, y_filtered)

    # imputating the missing values
    X_missing = X_full.copy()
    X_missing[np.where(missing_samples)[0], missing_features] = 0
    y_missing = y_full.copy()
    score_missing = estimateMissing(X_missing, y_missing)


if __name__ == '__main__':
    test()
