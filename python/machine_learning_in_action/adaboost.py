'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Classification - ada boost
# Author: Kelly Chan
# Date: Jan 25 2014
------------------------------------------------------
'''


'''

Function Tree

- adaClassifier
     |
     |----------- adaBoost
     |              |---------- bestStumpClassifier  <----- dataLoad
     |                                  |
     |----------- stumpPredicted -------|

'''

from numpy import *

# dataLoad
# (data, categories) return data, categories from rawData
def dataLoad():
    data = matrix([[ 1. , 2.1],
                   [ 2. , 1.1],
                   [ 1.3, 1. ],
                   [ 1. , 1. ],
                   [ 2. , 1. ]])
    categories = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data, categories


# stumpPredicted
# (predictedMatrix) return prdictedMatrix by comparing threshValue
def stumpPredicted(dataMatrix, col, threshValue, threshold):
    predictedMatrix = ones((shape(dataMatrix)[0],1))
    if threshold == 'lt':
        predictedMatrix[dataMatrix[:, col] <= threshValue] = -1.0
    else:
        predictedMatrix[dataMatrix[:, col] >  threshValue] = -1.0
    return predictedMatrix

# bestStumpClassifier
# (bestStump, minError, bestClassifiedMatrix)
# return bestStump, minError, bestClassifiedMatrix by stumpPredicted
def bestStumpClassifier(data, categories, weightedMatrix):

    # converting data to matrix
    dataMatrix = mat(data)
    categoryMatrix = mat(categories).T
    rows, cols = shape(dataMatrix)

    # calculating bestStump, minError, bestClassifiedMatrix
    steps = 10.0
    bestStump = {}
    minError = inf
    bestClassifiedMatrix = mat(zeros((rows,1)))
    for col in range(cols):
        rangeMin = dataMatrix[:, col].min()
        rangeMax = dataMatrix[:, col].max()
        stepSize = (rangeMax - rangeMin) / steps
        for step in range(-1, int(steps)+1):
            for comparison in ['lt', 'gt']:
                thresholdValue = rangeMin + float(step) * stepSize
                predictedMatrix = stumpPredicted(dataMatrix, col, thresholdValue, comparison)
                errorMatrix = mat(ones((rows,1)))
                errorMatrix[predictedMatrix == categoryMatrix] = 0
                weightedError = weightedMatrix.T * errorMatrix
                print "split: col %d, threshValue: %.2f, threshold: %s, weigthedError: %.3f" % (col, thresholdValue, comparison, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassifiedMatrix = predictedMatrix.copy()
                    bestStump['col'] = col
                    bestStump['thresholdValue'] = thresholdValue
                    bestStump['threshold'] = comparison
    return bestStump, minError, bestClassifiedMatrix



# adaBoost
# (weakClassifiedMatrix) return weakClassifiedMatrix by looping bestStumpClassifier
def adaBoost(data, categories, loops = 40): 

    # initialization
    rows = shape(data)[0]
    aggregatedClassifiedMatrix = mat(zeros((rows,1)))
    weightedMatrix = mat(ones((rows, 1))/rows) # probability

    weakClassifiers = []
    for i in range(loops):

        # (bestStump, minError, bestClassifiedMatrix) 
        # calculating bestStumpClassifier by data, categories, weightedMatrix
        bestStump, minError, bestClassifiedMatrix = bestStumpClassifier(data, categories, weightedMatrix)

        # calculating alpha
        alpha = float(0.5 * log((1.0-minError) / max(minError, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassifiers.append(bestStump)

        # printing weightedMatrix of this loop
        print "weightedMatrix: ", weightedMatrix.T
        # updating weightedMatrix for next loop
        alphaMatrix = multiply(-1 * alpha * mat(categories).T, bestClassifiedMatrix)
        weightedMatrix = multiply(weightedMatrix, exp(alphaMatrix))
        weightedMatrix = weightedMatrix / weightedMatrix.sum()

        # printing the bestClassifiedMatrix from bestStumpClassifier
        print "bestClassifiedMatrix: ", bestClassifiedMatrix.T
        # calculating aggregatedClassifiedMatrix
        aggregatedClassifiedMatrix += alpha * bestClassifiedMatrix
        print "aggregatedClassifiedMatrix: ", aggregatedClassifiedMatrix.T

        # calculating errorRate
        aggregatedErrors = multiply(sign(aggregatedClassifiedMatrix) != mat(categories).T, ones((rows, 1)))
        errorRate = aggregatedErrors.sum() / rows
        print "errorRate: ", errorRate, "\n"

        # if errorRate == 0.0, exit the loop
        if errorRate == 0.0:
            break
    return weakClassifiers


# adaClassifier
# (sign(aggregatedClassifiedMatrix)) return the category predicted
def adaClassifier(dataTest, weakClassifiers):
    dataTestMatrix = mat(dataTest)
    rows = shape(dataTestMatrix)[0]
    aggregatedClassifiedMatrix = mat(zeros((rows, 1)))
    for i in range(len(weakClassifiers)):
        predictedMatrix = stumpPredicted(dataTestMatrix, weakClassifiers[i]['col'], 
                                                         weakClassifiers[i]['thresholdValue'], 
                                                         weakClassifiers[i]['threshold'])
        aggregatedClassifiedMatrix += weakClassifiers[i]['alpha'] * predictedMatrix
        print "aggregatedClassifiedMatrix[%d]:" % i
        print aggregatedClassifiedMatrix

    return sign(aggregatedClassifiedMatrix)


#----------------------------------------------------------------
# testing

data, categories = dataLoad()
#print data, categories

weightedMatrix = mat(ones((5,1))/5)
#print bestStumpClassifier(data, categories, weightedMatrix)

weakClassifiers = adaBoost(data, categories,9)
print adaClassifier([[5,5],[0,0]], weakClassifiers)
