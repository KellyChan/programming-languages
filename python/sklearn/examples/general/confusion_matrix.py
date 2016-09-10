#-----------------------------------------------------------#
# Project: Confusion Matrix
# Author: Kelly Chan
# Date: Apr 22 2014
#-----------------------------------------------------------#

print(__doc__)

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import pylab as pl

def loadData():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y

def splitData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test

def createSVM():
    classifier = svm.SVC(kernel='linear')
    return classifier

def classify(classifier, X_train, y_train, X_test):
    clf = classifier.fit(X_train, y_train)
    yHat = clf.predict(X_test)
    return yHat

def computeConfusionMatrix(y_test, yHat):
    cm = confusion_matrix(y_test, yHat)
    return cm

def plotMatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    pl.show()

def test():

    X, y = loadData()
    X_train, X_test, y_train, y_test = splitData(X, y)

    classifier = createSVM()
    yHat = classify(classifier, X_train, y_train, X_test)
    cm = computeConfusionMatrix(y_test, yHat)
    print(cm)

    plotMatrix(cm)



if __name__ == '__main__':
    test()
