'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Regression - tree-based regression
# Author: Kelly Chan
# Date: Jan 27 2014
------------------------------------------------------
'''

'''

Prediction
1. forecast1: target tree
2. forecast2: model predicted - y = w0x0 + w1x1
3. forecast3: regression - weights


forecast1: target tree
- forecast
   |-------- (dataLoad)
   |-------- (treeCreated)  <---- targetMean / targetError
   |-------- treeForecast
                 |--------- regressionEvaluation


forecast2: model predicted - y = w0x0 + w1x1
- forecast
   |-------- (dataLoad)
   |-------- (treeCreated)  <---- regressionLeaf / regressionError
   |-------- treeForecast
                 |--------- modelEvaluation


forecast3: regression - weights
- regression
      |------ (dataLoad)


#------------------------------------------------------------------
# Master Function Tree

- forecast
   |
   |-------- (dataLoad)
   |
   |-------- (treeCreated) 
   |              |----------- targetMean / targetError
   |              |----------- regressionLeaf / regressionError
   |
   |-------- treeForecast
                 |--------- regressionEvaluation <---- targetMean / targetError
                 |--------- modelEvaluation      <---- regressionLeaf / regressionError




# creating tree from trainData
# tree structure
# (feature) featureIndex, featureValue
# (target/weights) leftNode, rightNode
#
# treePrune: to check if tree can be pruned

- treeCreated
   |------------- dataLoad
   |------------- dataSplit   
   |------------- bestFeatureSplit
   |                    |------------- targetMean  <---- regressionLeaf  <---- regression
   |                    |------------- targetError <---- regressionError <---- regression
   |                    |------------- dataSplit
   |
- treePrune
      |------- isTree
      |------- treeMean
      |------- dataSplit


'''

from numpy import *

# dataLoad
# (data) extracting data from raw file
def dataLoad(dataFile):
    data = []
    rawData = open(dataFile)
    for line in rawData.readlines():
        thisLine = line.strip().split('        ')
        filtedLine = map(float,thisLine)
        data.append(filtedLine)
    return data


# dataSplit
# (dataGt, dataLE) return dataGt, dataLE by featureValue in a specific featureIndex
# dataGt: >  featureValue
# dataLE: <= featureValue
def dataSplit(data, featureIndex, featureValue):
    dataGt = data[nonzero(data[:, featureIndex] >  featureValue)[0], :][0]
    dataLE = data[nonzero(data[:, featureIndex] <= featureValue)[0], :][0]
    return dataGt, dataLE



# targetMean
# retrun mean value of target
def targetMean(data):
    return mean(data[:,-1])  # data[:,-1]: get the last col

# targetError
# return variance of target
def targetError(data):
    return var(data[:,-1]) * shape(data)[0]

# bestTargetSplit
# (bestFeatureIndex, bestFeatureValue) 
# return bestFeatureIndex, bestFeatureValue by errorType=targetError 
#                                           or  leafType=targetMean
def bestFeatureSplit(data, leafType=targetMean, errorType=targetError, ops=(1,4)):

    # (featureIndex, featureValue)
    if len(set(data[:,-1].T.tolist()[0])) == 1:
        return None, leafType(data)  # featureIndex, featureValue

    bestTolerance = inf
    fixedTolerance = ops[0]  # tolerance on the error reduction
    tolerance = errorType(data)

    minN = ops[1]  # the minimum data instances to include in a split
    rows, cols = shape(data)

    bestFeatureIndex = 0
    bestFeatureValue = 0
    for featureIndex in range(cols-1):
        for featureValue in set(data[:, featureIndex]):
            dataGt, dataLE = dataSplit(data, featureIndex, featureValue)
            if (shape(dataGt)[0] < minN) or (shape(dataLE)[0] < minN):
                    continue
            newTolerance = errorType(dataGt) + errorType(dataLE)
            if newTolerance < bestTolerance:
                bestFeatureIndex = featureIndex
                bestFeatureValue = featureValue
                bestTolerance = newTolerance
    
    # (featureIndex, featureValue)
    if (tolerance - bestTolerance) < fixedTolerance:
        return None, leafType(data)   # featureIndex, featureValue

    # (featureIndex, featureValue)
    dataGt, dataLE = dataSplit(data, bestFeatureIndex, bestFeatureValue)
    if (shape(dataGt)[0] < minN) or (shape(dataLE)[0] < minN):
        return None, leafType(data)   # featureIndex, featureValue

    return bestFeatureIndex, bestFeatureValue



# treeCreated
# (tree) return tree by bestFeatureSplit
def treeCreated(data, leafType=targetMean, errorType=targetError, ops=(1,4)):

    # (bestFeatureIndex, bestFeatureValue)
    bestFeatureIndex, bestFeatureValue = bestFeatureSplit(data, leafType, errorType, ops)
    if bestFeatureIndex == None:
        return bestFeatureValue

    # constructing tree
    tree = {}
    tree['featureIndex'] = bestFeatureIndex
    tree['featureValue'] = bestFeatureValue
    dataGt, dataLE = dataSplit(data, bestFeatureIndex, bestFeatureValue)
    tree['leftNode'] = treeCreated(dataGt, leafType, errorType, ops)
    tree['rightNode'] = treeCreated(dataLE, leafType, errorType, ops)
    return tree



# isTree
# checking if type(obj) == 'dict'?
def isTree(obj):
    return (type(obj).__name__=='dict')


# treeMean
# return the mean of a tree by recursing left and right nodes
# treeMean = (leftValue + rightValue) / 2
def treeMean(tree):
    if isTree(tree['rightNode']):
        tree['rightNode'] = treeMean(tree['rightNode'])
    if isTree(tree['leftNode']):
        tree['leftNode'] = treeMean(tree['leftNode'])
    return (tree['leftNode'] + tree['rightNode']) / 2.0


# treePrune
# (tree) return tree after pruning
def treePrune(tree, testData):

    # if no testData, return treeMean
    if shape(testData)[0] == 0:
        return treeMean(tree)

    # if left/right node existed in the tree, 
    # get testDataGt, testDataLE by splitting testData with featureIndex, featureValue
    if (isTree(tree['rightNode']) or isTree(tree['leftNode'])):
        testDataGt, testDataLE = dataSplit(testData, tree['featureIndex'], tree['featureValue'])
    # recursing the subset(s) of left/right nodes
    if isTree(tree['leftNode']):
        tree['leftNode'] = treePrune(tree['leftNode'], testDataGt)
    if isTree(tree['rightNode']):
        tree['rightNode'] = treePrune(tree['rightNode'], testDataLE)

    # if no left/right nodes in a tree, calculating the error
    if not isTree(tree['leftNode']) and not isTree(tree['rightNode']):
        testDataGt, testDataLE = dataSplit(testData, tree['featureIndex'], tree['featureValue'])
        # error in left/right nodes
        # error = (leftTarget - leftNode)^2 + (rightTarget - rightNode)^2
        errorNotMerge = sum(power(testDataGt[:, -1] - tree['leftNode'], 2)) + \
                        sum(power(testDataLE[:, -1] - tree['rightNode'], 2))
        # error in left/right nodes combined
        # error = testTarget - (1/2 * (leftNode + rightNode))
        thisTreeMean = (tree['leftNode'] + tree['rightNode']) / 2.0
        errorMerge = sum(power(testData[:, -1] - thisTreeMean, 2))

        # if errorMerge < errorNotMerge, than merging the data
        if errorMerge < errorNotMerge:
            print "merging"
            return thisTreeMean
        else:
            return tree
    else:
        return tree

#-------------------------------------------------------------------------
# Regression


# regression
# (weightsMatrix, xMatrix, yMatrix)
# return model items by linear regression
def regression(data):
    rows, cols = shape(data)
    xMatrix = mat(ones((rows, cols)))
    yMatrix = mat(ones((rows, 1)))

    xMatrix[:, 1:cols] = data[:, 0:cols-1] # col 0: w0x0
    yMatrix = data[:, -1]
    
    xTx = xMatrix.T * xMatrix
    if linalg.det(xTx) == 0.0:
        raise NameError("This matrix is singular, it cannot do inverse.\n Try increasing the second value of ops.")
    weightsMatrix = xTx.I * (xMatrix.T * yMatrix)
    return weightsMatrix, xMatrix, yMatrix


# regressionLeaf
# (weightsMatrix) return weights of left/right nodes
def regressionLeaf(data):
    weightsMatrix, xMatrix, yMatrix = regression(data)
    return weightsMatrix


# regressionError
# return error of left/right nodes
# error = (yMatrix - yHat)^2
def regressionError(data):
    weightsMatrix, xMatrix, yMatrix = regression(data)
    yHat = xMatrix * weightsMatrix
    return sum(power(yMatrix - yHat, 2))

#-------------------------------------------------------------------------
# forecast


# regressionEvaluation
# (node) return the value of left/right node
def regressionEvaluation(node, testX):
    return float(node)

# modelEvaluation
# (xMatrix * node) retrun yHat with left/right node
def modelEvaluation(node, testX):
    cols = shape(testX)[1]
    xMatrix = mat(ones((1, cols+1)))
    xMatrix[:, 1:cols+1] = testX
    return float(xMatrix * node)


# treeForecast
# (yHat[i]) retrun yHat by each row
def treeForecast(tree, testX, testTarget=regressionEvaluation):

    if not isTree(tree):
        return testTarget(tree, testX)
    
    if testX[tree['featureIndex']] > tree['featureValue']:
        if isTree(tree['leftNode']):
            return treeForecast(tree['leftNode'], testX, testTarget)
        else:
            return testTarget(tree['leftNode'], testX)
    else:
        if isTree(tree['rightNode']):
            return treeForecast(tree['rightNode'], testX, testTarget)
        else:
            return testTarget(tree['rightNode'], testX)


# forecast
# (yHat) return yHat by searching treeForecast
def forecast(tree, testFeature, testTarget=regressionEvaluation):
    rows = len(testFeature)
    yHat = mat(zeros((rows,1)))
    for row in range(rows):
        yHat[row,0] = treeForecast(tree, mat(testFeature[row]), testTarget)
    return yHat
    


#---------------------------------------------------
# testing

dataFile = "G:\\vimFiles\\python\\examples\\machine_learning_in_action\\dataTreeRegression.txt"
data = dataLoad(dataFile)
data = mat(data)
# print treeCreated(data)


trainFile = "G:\\vimFiles\\python\\examples\\machine_learning_in_action\\dataTreePrune.txt"
testFile = "G:\\vimFiles\\python\\examples\\machine_learning_in_action\\dataTreePruneTest.txt"

trainData = dataLoad(trainFile)
trainData = mat(trainData)

testData = dataLoad(testFile)
testData = mat(testData)

#tree = treeCreated(trainData, ops=(0,1))
#treePruned = treePrune(tree, testData)
#print "tree: ", tree,"\n"
#print "treePruned: ", treePruned


#---------------------------------------------------
# testing - BikeSpeed

trainFile = "G:\\vimFiles\\python\\examples\\machine_learning_in_action\\dataBikeSpeedTrain.txt"
testFile = "G:\\vimFiles\\python\\examples\\machine_learning_in_action\\dataBikeSpeedTest.txt"

trainData = dataLoad(trainFile)
trainData = mat(trainData)

testData = dataLoad(testFile)
testData = mat(testData)


# Option1: (treeCreated, forecast)
print "----------------------------------"
print "Option1: (treeCreated, forecast)"
print "----------------------------------"
tree = treeCreated(trainData, ops=(1,20))
print tree
yHat = forecast(tree, testData[:,0]) # testData[:,0] feature
coef = corrcoef(yHat, testData[:,1],rowvar=0)[0,1] # testData[:,1] target
print "corrcoef1:",coef


# Option2: (treeCreated, regressionLeaf, regressionError, modelEvaluation)
print "------------------------------------------------------------------------"
print "Option2: (treeCreated, regressionLeaf, regressionError, modelEvaluation)"
print "------------------------------------------------------------------------"
tree = treeCreated(trainData, leafType = regressionLeaf, \
                              errorType = regressionError, ops=(1,20))
print tree
yHat = forecast(tree, testData[:,0], testTarget=modelEvaluation)
coef = corrcoef(yHat, testData[:,1],rowvar=0)[0,1]
print "corrcoef2:",coef

# Option3: (regression)
print '\n'
print "---------------------"
print "Option3: regression"
print "---------------------"
weights, xMatrix, yMatrix = regression(trainData)
print "weights:\n",weights
# y = w0x0 + w1x1
for row in range(shape(testData)[0]):
    yHat[row] = testData[row,0] * weights[1,0] + weights[0,0]
coef = corrcoef(yHat, testData[:,1], rowvar=0)[0,1]
print "corrcoef3:",coef
