import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from tensorflow.contrib import skflow


def load_cifar(file):
    with open(file, 'rb') as inf:
        cifar = pickle.load(inf, encoding='latin1')
    data = cifar['data'].reshape((10000, 3, 32, 32))
    data = np.rollaxis(data, 3, 1)
    data = np.rollaxis(data, 3, 1)
    y = np.array(cifar['labels'])
    
    mask = (y == 2) | (y == 9)
    data = data[mask]
    y = y[mask]

    return data, y


def classifier_svm(digits):
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(digits.data, digits.target)
    predicted = classifier.predict(digits.data)
    print(np.mean(digits.target == predicted))


def classifier_tf(digits):
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)
    n_classes = len(set(y_train))
    classifier = skflow.TensorFlowLinearClassifier(n_classes=n_classes)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(metrics.classification_report(y_true=y_test, y_pred=y_pred))


if __name__ == '__main__':

    digits = load_digits()
    fig = plt.figure(figsize=(3,3))
    plt.imshow(digits['images'][66], cmap="gray", interpolation='none')
    plt.show()

    classifier_svm(digits)
    classifier_tf(digits)
