'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Classification - k-Nearest Neighbors
# Case: Digit Recognizer

# Author: Kelly Chan
# Date: Jan 20 2014
------------------------------------------------------
'''

# imgToVector
# return imgVector by converting image matrix
def imgToVector(filename):
    imgVector = zeros((1,1024))
    img = open(filename)
    for i in range(32):
        lineString = img.readline()
        for j in range(32):
            imgVector[0, 32*i+j] = int(lineString[j])
    return imgVector


# testDigitRecognizer
# return the digits of test data by computing knn with train data
def testDigitRecognizer(trainData, testData):

    # extracting labels, trainMatrix from train data
    labels = []
    rowsTrain = len(trainData)
    for i in range(rowsTrain):
        img = trainData[i]
        imgLabel = img.split('.')[0]
        imgLabel = int(imgLabel.split('_')[0])
        labels.append(imgLabel)
        trainMatrix[i,:] = imgToVector('trainDigits/%s' % img)

    # return test digits by knn algorithm
    error = 0.0
    rowsTest = len(testData)
    for i in range(rowsTest):
        img = testData[i]
        imgLabel = img.split('.')[0]
        imgLabel = int(imgLabel.split('_')[0])
        testX = imgToVector('testDIgits/%s' % img)
        output = knn(testX, trainMatrix, labels, 3)
        print "output: %d, the correct answer: %d" % (output, imgLabel)
        if (output != imgLabel):
            error += 1.0
    print "\ntotal # of errors: %d" % error
    print "\ntotal error rate: %f" % (error/float(rowsTest))



