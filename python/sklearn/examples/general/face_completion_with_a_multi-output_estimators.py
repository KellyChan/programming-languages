#---------------------------------------------------------------#
# Project: Face completion with a multi-output estimators
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV


def loadData():
    data = fetch_olivetti_faces()
    targets = data.target
    return data, targets

def splitData(data, targets):
    data = data.images.reshape((len(data.images), -1))
    train = data[targets < 30]
    test = data[targets >= 30]  # test on independent people
    return data, train, test

def testSubset(test):
    n_faces = 5

    rng = check_random_state(4)
    face_ids = rng.randint(test.shape[0], size=(n_faces, ))
    test = test[face_ids, :]

    return test, n_faces

def sampling(data, train, test):
    n_pixels = data.shape[1]
    X_train = train[:, :np.ceil(0.5 * n_pixels)]  # upper half of the faces
    y_train = train[:, np.floor(0.5 * n_pixels):]  # lower half of the faces
    X_test = test[:, :np.ceil(0.5 * n_pixels)]
    y_test = test[:, np.floor(0.5 * n_pixels):]
    return X_train, y_train, X_test, y_test

def createClassifiers():

    ESTIMATORS = {"Extra trees": ExtraTreesRegressor(n_estimators=10, \
                                                     max_features=32, \
                                                     random_state=0), \
                  "K-nn": KNeighborsRegressor(), \
                  "Linear regression": LinearRegression(), \
                  "Ridge": RidgeCV()
                 }

    return ESTIMATORS

def classify(ESTIMATORS, X_train, y_train, X_test):

    y_test_predict = dict()

    for name, estimator in ESTIMATORS.items():
        estimator.fit(X_train, y_train)
        y_test_predict[name] = estimator.predict(X_test)

    return y_test_predict


def plotPredict(ESTIMATORS, n_faces, X_test, y_test, y_test_predict):
    
    # plot the completed faces
    image_shape = (64, 64)

    n_cols = 1 + len(ESTIMATORS)

    plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
    plt.suptitle("Face completion with multi-output estimators", size=16)

    for i in range(n_faces):
        true_face = np.hstack((X_test[i], y_test[i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title='true faces')

        sub.axis("off")
        sub.imshow(true_face.reshape(image_shape), \
                   cmap=plt.cm.gray, \
                   interpolation='nearest')

        for j, est in enumerate(sorted(ESTIMATORS)):
            completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

            if i:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)
            else:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j, title=est)

            sub.axis("off")
            sub.imshow(completed_face.reshape(image_shape), \
                       cmap=plt.cm.gray, \
                       interpolation='nearest')
    plt.show()
            


def test():
    
    data, targets = loadData()
    data, train, test = splitData(data, targets)
    test, n_faces = testSubset(test)

    X_train, y_train, X_test, y_test = sampling(data, train, test)

    ESTIMATORS = createClassifiers()
    y_test_predict = classify(ESTIMATORS, X_train, y_train, X_test)

    plotPredict(ESTIMATORS, n_faces, X_test, y_test, y_test_predict)



if __name__ == '__main__':
    test()
