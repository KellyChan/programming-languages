'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Classification - k-Nearest Neighbors
# Author: Kelly Chan
# Date: Jan 20 2014
------------------------------------------------------
'''


# DataToMatrix
# return dataMatrix, labels extracting from data file
def DataToMatrix(filename):

    # shaping an empty dataMatrix
    data = open(filename)
    n = len(data.readlines())
    dataMatrix = zeros((n,3))

    # extracting dataMatrix, labels from data file
    index = 0
    labels = []
    data = open(filename)
    for line in data.readlines():
        line = line.strip().split('\t')
        dataMatrix[index, :] = line[0:3]
        labels.append(int(line[-1]))
        index += 1
    return dataMatrix, labels


# AutoNorm
# return normalized matrix (scale: 0-1) by (value - min) / (max - min)
def AutoNorm(dataMatrix):

    # calculating range matrix by columns: max - min
    colsMin = dataMatrix.min(0)
    colsMax = dataMatrix.max(0)
    colsRange = colsMax - colsMin

    # normalizing data by cells: (value - min) / range
    rows = dataMatrix.shape[0]
    normMatrix = zeros(shape(dataMatrix))
    normMatrix = dataMatrix - tile(colsMin, (rows, 1))
    normMatrix = normMatrix / tile(colsRange, (rows, 1))

    return normMatrix, colsRange, colsMin



# (classification) kNN
# return the nearest label in k neighbors by computing Euclidean Distance
def knn(testX, trainData, labels, k):
    
    # (testX, trainData) computing Euclidean Distance
    n = trainData.shape[0]
    distanceMatrix = tile(testX, (n,1)) - trainData  # tile: [testX]_n
    distanceMatrix = distanceMatrix**2
    distances = distanceMatrix.sum(axis=1) # axis=0: by cols | aisx=1: by rows
    distances = distances**0.5
    distancesIndex = distances.argsort()  # argsort(): index by ascending values


    # (k, labels) return the nearest label in k neighbors
    kDistances = {}
    for i in range(k):
        label = labels[distancesIndex[i]]
        # counting # of label in k values, .get(key, value), default = 0
        kDistances[label] = kDistances.get(label,0) + 1
    # .iteritems: loop keys, operator.itemgetter(1): sort by values, descending
    kDistances = sorted(kDistances.iteritems(), key=operator.itemgetter(1), reverse = True)
    return kDistances[0][0]



# knnTest
# function testing: FileToMatrix, AutoNorm, knn
def knnTest(datafile):

    # converting data matrix, normalizing
    dataMatrix, labels = FileToMatrix(datafile)
    normMatrix, colsRange, colsMin = AutoNorm(dataMatrix)

    error = 0.0
    splitRatio = 0.10
    rows = normMatrix.shape[0]
    splitIndex = int(rows * splitRatio)

    for i in range(splitIndex):
        # return values by knn algorithm
        output = knn(normMatrix[i,:], normMatrix[splitIndex:rows, :], labels[splitIndex:rows], 3)
        print "output: %d, the correct answer: %d" % (output, labels[i])

        if (output != labels[i]): 
            error += 1.0
    print "the total error rate: %f" % (error/float(splitIndex))


#---------------------------------------------------------------
# unit test

from numpy import *
import operator

group = array([[1.0,1.1],
               [1.0,1.0],
               [0,0],
               [0,0.1]])
labels = ['A',
          'A',
          'B',
          'B']

print knn([0.5,0.5], group, labels, 2)


