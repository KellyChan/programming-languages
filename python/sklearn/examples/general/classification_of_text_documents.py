#---------------------------------------------------------------#
# Project: Classification of text documents: using a MLComp dataset
# Author: Kelly Chan
# Date: Apr 24 2014
#---------------------------------------------------------------#

from __future__ import print_function

import sys
import os
from time import time

import numpy as np
import pylab as pl
import scipy.sparse as sp

from sklearn.datasets import load_mlcomp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

print(__doc__)

if 'MLCOMP_DATASETS_HOME' not in os.environ:
    print("MLCOMP_DATASETS_HOME not set; please follow the above instructions")
    sys.exit(0)


def loadTrain():
    print("Loading 20 newsgroups training set...")
    news_train = load_mlcomp('20news-18828', 'train')
    
    print(news_train.DESCR)
    print("%d documents" % len(news_train.filenames))
    print("%d categories" % len(news_train.target_names))

    return news_train

def loadTest():

    print("Loading 20 newsgroups test set...")
    news_test = load_mlcomp('20news-18828', 'test')

    t0 = time()
    print("done in %fs" % (time() - t0))
    print("%d documents" % len(news_test.filenames))
    print("%d categories" % len(news_test.target_names))

    return news_test


def extractFeatures(news_train, news_test):

    print("Extracting features from the dataset using a sparse vertorizer")
    t0 = time()
    vectorizer = TfidfVectorizer(encoding='latin1')
    X_train = vectorizer.fit_transform((open(f).read() \
                                                for f in news_train.filenames))
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    assert sp.issparse(X_train)

    y_train = news_train.target

    
    print("Predicting the labels of the test set...")
    print("Extracting features from the dataset using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform((open(f).read() \
                                           for f in news_test.filenames))
    y_test = news_test.target

    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X_test.shape)

    return X_train, X_test, y_train, y_test


def benchmark(clf_class, params, name, X_train, X_test, y_train, y_test):

    print("parameters", params)
    t0 = time()
    clf = clf_class(**params).fit(X_train, y_train)
    print("done in %fs" % (time() - t0))

    if hasattr(clf, 'coef_'):
        print("Percentage of non zeros coef: %f" \
                % (np.mean(clf.coef_ != 0) * 100))


    print("Predicting the outcomes of the testing set")
    t0 = time()
    pred = clf.predict(X_test)
    print("done in %fs" % (time() - t0))


    print("Classification report on test set for classifier:")
    print(clf)
    print()
    print(clasification_report(y_test, \
                               pred, \
                               target_names=news_test.target_names))

    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:")
    print(cm)

    # show confusion matrix
    pl.matshow(cm)
    pl.title('Confusion matrix of the %s classifier' % name)
    pl.colorbar()

    pl.show()


def testBenchmark(X_train, X_test, y_train, y_test):

    print("Testbenching a linear classifier...")
    parameters = { 'loss': 'hinge', \
                   'penalty': '12', \
                   'n_iter': 50, \
                   'alpha': 0.00001, \
                   'fit_intercept': True, \
                 }

    benchmark(SGDClassifier, parameters, 'SGD', X_train, X_test, y_train, y_test)


    print("Testbenching a MultinomialNB classifier...")
    parameters = {'alpha': 0.01}
    benchmark(MultinomicalNB, parameters, 'MultinomialNB', X_train, X_test, y_train, y_test)


def main():

    news_train = loadTrain()
    news_test = loadTest()

    X_train, X_test, y_train, y_test = extractFeatures(news_train, news_test)

    testBenchmark(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
