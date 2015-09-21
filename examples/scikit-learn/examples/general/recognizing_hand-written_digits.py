#---------------------------------------------------------------#
# Project: Recognizing hand-written digits
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

print(__doc__)

import pylab as pl
from sklearn import datasets, svm, metrics

def loadData():
    digits = datasets.load_digits()
    return digits

def loadPic(digits):
    for index, (image, label) in enumerate(zip(digits.images, digits.target)[:4]):
        pl.subplot(2, 4, index+1)
        pl.axis('off')
        pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
        pl.title('Training: %i' % label)


def reshapeData(digits):
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, n_samples

def createSVM():
    classifier = svm.SVC(gamma=0.001)
    return classifier


def classify(classifier, digits, data, n_samples):

    expected = digits.target[n_samples/2: ]

    classifier.fit(data[:n_samples/2], digits.target[:n_samples/2])
    predicted = classifier.predict(data[n_samples/2:])

    print("Classification report for classifier %s: \n%s\n" \
            % (classifier, metrics.classification_report(expected, predicted)))

    print("Confusion matrix:\n%s" \
            % metrics.confusion_matrix(expected, predicted))

    return expected, predicted


def plotPredict(digits, predicted, n_samples):
    for index, (image, prediction) in enumerate(\
            zip(digits.images[n_samples/2:], predicted)[:4]):
        pl.subplot(2, 4, index+5)
        pl.axis('off')
        pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
        pl.title('Prediction %i' % prediction)
    pl.show()


def test():
    digits = loadData()
    loadPic(digits)

    data, n_samples = reshapeData(digits)

    classifier = createSVM()
    expected, predicted = classify(classifier, digits, data, n_samples)
    plotPredict(digits, predicted, n_samples)



if __name__ == '__main__':
    test()



