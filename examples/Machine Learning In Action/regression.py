'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Regression - linear regression
# Author: Kelly Chan
# Date: Jan 25 2014
------------------------------------------------------
'''

from numpy import *


# dataLoad
# (features, target) extracting features, target from rawData
def dataLoad(dataFile):
    features = []
    target = []
    nFeatures = len(open(dataFile).readline().split('\t')) - 1
    rawData = open(dataFile)
    for line in rawData.readlines():
        thisLine = line.strip().split('\t')
        thisFeatures = []
        for i in range(nFeatures):
            thisFeatures.append(float(thisLine[i]))
        features.append(thisFeatures)
        target.append(float(thisLine[-1]))
    return features, target


# regressionWeights
# (weightsMartix) retrun weightsMatrix by calculating w = (X^T * X)^(-1) * X^T * Y
def regressionWeights(features, target):
    xMatrix = mat(features)
    yMatrix = mat(target).T
    # X^T * X
    xTx = xMatrix.T * xMatrix
    # linalg.det(): computing the determinate
    if linalg.det(xTx) == 0.0:
        print "The matrix is singular, could not do inverse."
        return
    # w = (X^T * X)^(-1) * X^T * Y
    weightsMatrix = xTx.I * xMatrix.T * yMatrix
    return weightsMatrix




dataFile = "G:\\vimFiles\\python\\examples\\machine_learning_in_action\\dataRegression.txt"
features, target = dataLoad(dataFile)

weightsMatrix = regressionWeights(features, target)

featuresMatrix = mat(features)
targetMatrix = featuresMatrix * weightsMatrix
#print targetMatrix

# IMPORTANCE: checking the rawTarget and the predictedTarget if match by corrcoef
print corrcoef(targetMatrix.T, target)


