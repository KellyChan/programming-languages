'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Regression - linear regression
# Author: Kelly Chan
# Date: Jan 25 2014
------------------------------------------------------
'''


'''

Function Tree

- regressionWeights
     |-------------- dataLoad


- squaredError
   |
- regressionLocalWeightsTest
      |
      |--------- regressionLocalWeights
      |--------- dataLoad


NOTE: which lambda should be chosen?
NOTE: Normalizing rawData before implementing regressionRidge
- regressionRidgeWeightsTest
      |--------- regressionRidge
      |--------- dataLoad


# Lasso
- regressionStagewise
      |--------- regularize
      |--------- squaredError
'''


from numpy import *


# dataLoad
# (features, target) extracting features, target from rawData
def dataLoad(dataFile):
    features = []
    target = []
    nFeatures = len(open(dataFile).readline().split('        ')) - 1
    rawData = open(dataFile)
    for line in rawData.readlines():
        thisLine = line.strip().split('        ')
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

#---------------------------------------------------------------------------------

# regressionLocalWeights
# (yHatTestX) return yHat by locally weighted linear regression
# kernel: kernel = exp (|xi - x|/(-2k^2))
# wHat = (X^T * kernel * X)^(-1) * X^T * kernel * y
# yHatTestX = testX * wHat
def regressionLocalWeights(testX, features, target, k = 1.0):
    xMatrix = mat(features)
    yMatrix = mat(target).T
    rows = shape(xMatrix)[0]
    kernelMatrix = mat(eye((rows)))
    for row in range(rows):
        differenceMatrix = testX - xMatrix[row, :]
        # kernel: wHat = exp (|xi - x|/(-2k^2))
        kernelMatrix[row, row] = exp(differenceMatrix * differenceMatrix.T / (-2.0*k**2))
        xTx = xMatrix.T * (kernelMatrix * xMatrix)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, it cannot do inverse."
        return
    # wHat = (X^T * kernel * X)^(-1) * X^T * kernel * y
    weightsMatrix = xTx.I * (xMatrix.T * kernelMatrix * yMatrix)
    return testX * weightsMatrix



# regressionLocalWeightsTest
# (yHat) return yHat by locally weighted linear regression
def regressionLocalWeightsTest(testFeatures, trainFeatures, trainTarget, k=1.0):
    rows = shape(testFeatures)[0]
    yHat = zeros(rows)
    for row in range(rows):
        yHat[row] = regressionLocalWeights(testFeatures[row],trainFeatures, trainTarget, k)
    return yHat

# error
# return the error of (y - yHat)^2
def squaredError(yArray,yHatArray):
    return ((yArray-yHatArray)**2).sum()

#---------------------------------------------------------------------------------



# regressionRidgeWeights
# (ridgeWeights) retrun ridgeWeights by calculating
# Y = X^T * X + lambda * I
# W = (X^T * X + lambda * I)^(-1) * X^T * y
def regressionRidgeWeights(xMatrix, yMatrix, lamb=0.2): # lamb: lambda 
    xTx = xMatrix.T * xMatrix
    # Y = X^T * X + lambda * I
    ridgeMatrix = xTx + eye(shape(xMatrix)[1])*lamb
    if linalg.det(ridgeMatrix) == 0.0:
        print "This matrix is singular, it cannot do inverse."
        return
    # W = (X^T * X + lambda * I)^(-1) * X^T * y
    ridgeWeights = ridgeMatrix.I * (xMatrix.T * yMatrix)
    return ridgeWeights


# regressionRidgeWeightsTest
# (weightsMatrix) return different weights with different lambda
#NOTE: which lambda should be chosen?
#NOTE: Normalizing the rawData before implementing Ridge Regression
def regressionRidgeWeightsTest(features, target):
    xMatrix = mat(features)
    yMatrix = mat(target).T

    # normalizing target
    yMeanMatrix = mean(yMatrix,0)
    yMatrix = yMatrix - yMeanMatrix

    # normalizing features
    xMeanMatrix = mean(xMatrix,0)
    xVar = var(xMatrix,0) # var: variance
    xMatrix = (xMatrix - xMeanMatrix) / xVar


    # get different weights by calculating different lambdas
    nLambdas = 30
    weightsMatrix = zeros((nLambdas, shape(xMatrix)[1]))
    for i in range(nLambdas):
        weights = regressionRidgeWeights(xMatrix, yMatrix, exp(i-10))
        weightsMatrix[i,:] = weights.T
    return weightsMatrix 


#---------------------------------------------------------------------------------

# regularize
# (xRegularizedMatrix) return xMatrix after regularization
def regularize(featuresMatrix):
    xMatrix = featuresMatrix.copy()
    xMeanMatrix = mean(xMatrix,0) # mean
    xVariance = var(xMatrix,0)  # variance
    xRegularizedMatrix = (xMatrix - xMeanMatrix) / xVariance
    return xRegularizedMatrix


# regressionStagewise
# (weightSet) return weightSet by calculating lasso for n loops
def regressionStagewise(features, target, eps=0.01, loops=100):
    xMatrix = mat(features)
    yMatrix = mat(target).T

    # normalizing yMatrix
    yMeanMatrix = mean(yMatrix,0)
    yMatrix =yMatrix - yMeanMatrix

    # normalizing xMatrix
    #(regularize) calling the function regularize to regularize xMatrix
    xMatrix = regularize(xMatrix)

    rows, cols = shape(xMatrix)
    weightsMatrix = zeros((cols,1))
    testWeightsMatrix = weightsMatrix.copy()
    maxWeightsMatrix = weightsMatrix.copy()

    weightSet = zeros((loops, cols))
    for loop in range(loops):
        #print weightsMatrix.T
        lowestError = inf
        for col in range(cols):
            for sign in [-1,1]:
                testWeightsMatrix = weightsMatrix.copy()
                # calculating weightsMatrix
                testWeightsMatrix[col] += eps * sign
                # Y = X * weights
                yTestMatrix = xMatrix * testWeightsMatrix
                rssError = squaredError(yMatrix.A, yTestMatrix.A)
                if rssError < lowestError:
                    lowestError = rssError
                    maxWeightsMatrix = testWeightsMatrix
        weightsMatrix = maxWeightsMatrix.copy()
        weightSet[loop,:] = weightsMatrix.T
    return weightSet



#---------------------------------------------------------------------------
# testing


#dataFile = "G:\\vimFiles\\python\\examples\\machine_learning_in_action\\dataRegression.txt"
#features, target = dataLoad(dataFile)

#weightsMatrix = regressionWeights(features, target)

#featuresMatrix = mat(features)
#yHat = featuresMatrix * weightsMatrix
#print targetMatrix

# IMPORTANCE: checking the rawTarget and the predictedTarget if match by corrcoef
#print corrcoef(yHat.T, target)


# locally weighted linear regression
#yHat = regressionLocalWeights(features[0], features, target, k=1.0)
#print regressionLocalWeights(features[0], features, target, k=0.001)
#print regressionLocalWeights(features[0], features, target, k=0.003)

#print  regressionLocalWeightsTest(features, features, target, k=1.0)

# printing error
#print error(array(target), array(yHat))



dataFile ="G:\\vimFiles\\python\\examples\\machine_learning_in_action\\dataRidgeRegression.txt"
features, target = dataLoad(dataFile)
weightsMatrix = regressionRidgeWeightsTest(features, target)


# plotting the different weights with different lambda
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(weightsMatrix)
#plt.show()


# Stagewise Regression
#print regressionStagewise(features, target, eps=0.01, loops=200)
#print regressionStagewise(features, target, eps=0.001, loops=5000)
