#---------------------------------------------------------------#
# Project: Comparing randomized search and grid search for hyperparameter estimation
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

print(__doc__)

import numpy as np

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

def loadData():
    iris = load_digits()
    X = iris.data
    y = iris.target
    return X, y

def createClassifier():
    clf = RandomForestClassifier(n_estimators=20)
    return clf

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0: .3f} (std: {1: .3f})".format(\
              score.mean_validation_score, \
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")



def createRandomSearch(clf, X, y):

    param_dist = {"max_depth": [3, None], \
                  "max_features": sp_randint(1, 11), \
                  "min_samples_split": sp_randint(1, 11), \
                  "min_samples_leaf": sp_randint(1, 11), \
                  "bootstrap": [True, False], \
                  "criterion": ["gini", "entropy"]
                 }

    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, \
                                            n_iter=n_iter_search)
    start = time()
    random_search.fit(X, y)

    print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)


def createParamGrid(clf, X, y):
    param_grid = {"max_depth": [3, None], \
                  "max_features": [1, 3, 10], \
                  "min_samples_split": [1, 3, 10], \
                  "min_samples_leaf": [1, 3, 10], \
                  "bootstrap": [True, False], \
                  "criterion": ['gini', 'entropy'] \
                 }

    
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    
    start = time()
    grid_search.fit(X, y)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings." \
            % (time() - start, len(grid_search.grid_scores_)))
    report(grid_search.grid_scores_)


def test():

    X, y = loadData()

    clf = createClassifier()
    createRandomSearch(clf, X, y)
    createParamGrid(clf, X, y)

if __name__ == '__main__':
    test()


