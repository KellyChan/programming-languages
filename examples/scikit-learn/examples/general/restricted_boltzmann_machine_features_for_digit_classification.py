#---------------------------------------------------------------#
# Project: Restricted Boltzmann Machine features for digit classification
# Author: Kelly Chan
# Date: Apr 24 2014
#---------------------------------------------------------------#


from __future__ import print_function

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from sklearn import datasets
from sklearn.cross_validation import train_test_split

from sklearn import metrics
from sklearn.pipeline import Pipeline

from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM



def nudgeData(X, Y):
    """

    This produces a dataset 5 times bigger than the original one,
    by moving the 8*8 images in X aroud by 1px to left, right, down, up
    """
    direction_vectors = [\
                          [[0, 1, 0], \
                           [0, 0, 0], \
                           [0, 0, 0]], \

                          [[0, 0, 0], \
                           [1, 0, 0], \
                           [0, 0, 0]], \
                          
                          [[0, 0, 0], \
                           [0, 0, 1], \
                           [0, 0, 0]], \
                          
                          [[0, 0, 0], \
                           [0, 0, 0], \
                           [0, 1, 0]] \
                        ]

    shift = lambda x, w: convolve(x.reshape((8, 8)), \
                                  mode='constant', \
                                  weights=w).ravel()

    X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) \
                                     for vector in direction_vectors])

    Y = np.concatenate([Y for _ in range(5)], axis=0)

    return X, Y


def loadData():
    digits = datasets.load_digits()
    X = np.asarray(digits.data, 'float32')
    X, Y = nudgeData(X, digits.target)
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

    return X, Y

def splitData(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, \
                                                        test_size=0.2, \
                                                        random_state=0)

    return X_train, X_test, Y_train, Y_test


def creatRBMLogistic():

    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    classifier = Pipeline(steps=[('rbm', rbm), \
                                 ('logistic', logistic)])


    return classifier, rbm, logistic

def createLogistic():
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    return logistic_classifier

def predictRBMLogistic(classifier, rbm, logistic, X_train, Y_train):

    """Hyper-parameters
    
    There were set by cross-validation using a GridSearchCV.
    Here we are not performing cross_validation to save time.
    """

    rbm.learning_rate = 0.06
    rbm.n_iter = 20

    # more components tend to give better prediction performance
    # but larger fitting time
    rbm.n_components = 100
    logistic.C = 6000.0

    # training RBM-Logistic Pipeline
    yHat = classifier.fit(X_train, Y_train)

    return yHat


def predictLogistic(logistic_classifier, X_train, Y_train):

    # training Logistic regression
    yHat = logistic_classifier.fit(X_train, Y_train)
    return yHat


def evaluate(classifier, logistic_classifier, X_test, Y_test):
    
    print()
    print("Logistic regression using RBM features:\n%s\n" \
              % (metrics.classification_report(Y_test, \
                                               classifier.predict(X_test))))

    print("Logistic regression using raw pixel features:\n%s\n" \
              % (metrics.classification_report(Y_test, \
                                               logistic_classifier.predict(X_test))))



def plotPics(rbm):

    plt.figure(figsize=(4.2, 4))

    for i, comp in enumerate(rbm.components_):
        plt.subplot(10, 10, i+1)
        plt.imshow(comp.reshape((8, 8)), \
                   cmap=plt.cm.gray_r, \
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('100 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()


def main():

    X, Y = loadData()
    X_train, X_test, Y_train, Y_test = splitData(X, Y)

    classifier, rbm, logistic = creatRBMLogistic()
    logistic_classifier = createLogistic()

    yHat_rbm = predictRBMLogistic(classifier, rbm, logistic, X_train, Y_train)
    yHat_log = predictLogistic(logistic_classifier, X_train, Y_train)

    evaluate(classifier, logistic_classifier, X_test, Y_test)
    plotPics(rbm)

if __name__ == '__main__':
    main()

