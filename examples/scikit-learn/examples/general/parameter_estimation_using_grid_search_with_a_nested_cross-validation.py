#---------------------------------------------------------------#
# Project: Parameter estimation using grid search with a nested cross-validation
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

print(__doc__)

def loadData():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target
    return X, y, n_samples

def splitData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.5, \
                                                        random_state=0)
    return X_train, X_test, y_train, y_test

def estimateParameters(X_train, X_test, y_train, y_test):

    tuned_parameters = [{'kernel': ['rbf'], \
                         'gamma': [1e-3, 1e-4], \
                         'C': [1, 10, 100, 1000]}, \
                        {'kernel': ['linear'], \
                         'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']
    for score in scores:

        print("# Tuning hyper-parameters for %s\n" % score)

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:\n")
        print(clf.best_estimator_)

        print("\nGrid scores on development set:\n")
        for params, mean_score, scores in clf.grid_scores_:
            print("%.3f (+/-%.03f) for %r" % (mean_score, scores.std() / 2, params))

        print("\nDetailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


def test():
    X, y, n_samples = loadData()
    X_train, X_test, y_train, y_test = splitData(X, y)
    estimateParameters(X_train, X_test, y_train, y_test)



# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.

if __name__ == '__main__':
    test()
