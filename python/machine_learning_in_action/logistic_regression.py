'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Classification - logistic regression
# Author: Kelly Chan
# Date: Jan 23 2014
------------------------------------------------------
'''

'''

Function Tree

- multiTesting
     |
- testing
     |------- modifiedStocGradientAscent
     |------- classifier
                  |----------- sigmoid
                                  |
                  |----------- modifiedStocGradientAscent  ----> bestFitPlot
                  |----------- (stochasticGradientAscent)  ----> bestFitPlot
                  |----------- (gradientAscent)            ----> bestFitPlot
                                   |---------- dataLoad -------------|                                                                   

'''


from numpy import *

# dataLoad
# (data, category) return data, category by spliting raw data
def dataLoad(dataPath):
    data = []
    category = []
    rawData = open(dataPath)
    for line  in rawData.readlines():
        attribute = line.strip().split('\t')
        # IMPORTANCE: appending dummyDelta value (1.0) for each row 
        data.append([1.0, float(attribute[0]), float(attribute[1])])
        category.append(int(attribute[2]))
    return data, category



# sigmoid
# (z) return sigmoid by calculating: 1/(1+e^-z)
# sigmoid(0)=0.5, 0 < sigmoid < 1
def sigmoid(z):
    return 1.0/(1+exp(-z))



# gradientAscent
# (weights) return col weights by calculating gradient ascent
# weights: maxDelta, attribute1, attribute2
# - maxDelta: how far to go
# - direction: (attribute1, attribute2)
# 
# gradientAscent: w = w + alpha * delta(f(w))
# gradientDescent: w = w - alpha * delta(f(w))
# delta(f(w)): delta = X^T * error
# error: error = categoryY - sigmoidX
# sigmoidX: sigmoidX = sigmoid(dataX * unitWeights)
def gradientAscent(data, category):

    dataMatrix = mat(data)
    categoryMatrix = mat(category).transpose()
    rows, cols = shape(dataMatrix)

    maxStep = 500  # stopping point
    alpha = 0.001  # step size towards the target
    weights = ones((cols,1)) # col weights
    for i in range(maxStep):
        # rows: converting dataMatrix to sigDataMatrix in scale (0,1)
        sigDataMatrix = sigmoid(dataMatrix * weights)
        # rows: category (0,1) - sigDataMatrix (0,1)
        error = categoryMatrix - sigDataMatrix
        # cols transposed to rows: calculating weights for each col
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# stochasticGradientAscent
# (weights) return weights by calculating stochastic Gradient Ascent
# 
# compared with gradientAscent
# sigmoid f(x): sum(data[i]*weights) <- dataMatrix * weights
# category: category[i] <- categoryMatrix[i]
#
# stochasticGradientAscent: vector, random processing, computing vectors one by one
# gradientAscent: matrix, batch processing, computing matrix for maxStep times
def stochasticGradientAscent(data, category):

    data = array(data)
    rows, cols = shape(data)

    alpha = 0.01
    weights = ones(cols)
    for i in range(rows):
        # IMPORTANCE: sum(data[i]*weights) <- dataMatrix * weights 
        sigData = sigmoid(sum(data[i]*weights))
        error = category[i] - sigData
        weights = weights + alpha * data[i] * error
    return weights


# modifiedStocGradientAscent
# return weights by random selection
#
# stochasticGradientAscent: alpha fixed, weights by each vector
# modifiedStocGradientAscent: alpha dynamic, weights by random vector, for n steps
def modifiedStocGradientAscent(data, category, steps=150):
   
    data = array(data)
    rows, cols = shape(data)
    weights = ones(cols)
    for step in range(steps):
        dataIndex = range(rows)  # get row index [0..n]
        for row in range(rows):
            # alpha improves the oscillations in the dataset
            # alpha decreased when row increased, alpha > 0
            alpha = 4 / (1.0 + step + row) + 0.01
            # get a randomIndex
            randomIndex = int(random.uniform(0, len(dataIndex)))

            # calculating weights by random vector
            sigData = sigmoid(sum(data[randomIndex]*weights))
            error = category[randomIndex] - sigData
            weights = weights + alpha * error * data[randomIndex]
            # removing randomIndex from dataIndex, no duplication
            del(dataIndex[randomIndex])

    return weights





# bestFitPlot
# plotting the best fit in (x,y) scatterplot  
def bestFitPlot(weightsMatrix, dataPath):
    import matplotlib.pyplot as plt

    data, category = dataLoad(dataPath)
    dataMatrix = array(data)
    weights = weightsMatrix.getA()  # .getA for matrix

    rows = shape(dataMatrix)[0]
    xCord1 = []; yCord1 = []  # category 1
    xCord2 = []; yCord2 = []  # category 0
    for i in range(rows):
        if int(category[i]) == 1:
            xCord1.append(dataMatrix[i,1]) # category1 attribute1
            yCord1.append(dataMatrix[i,2]) # category1 attribute2
        else:
            xCord2.append(dataMatrix[i,1]) # category0 attribute1
            yCord2.append(dataMatrix[i,2]) # category0 attribute2


    fig = plt.figure()
    ax = fig.add_subplot(111)
    # scatterplot
    ax.scatter(xCord1, yCord1, s=30, c='red', marker='s')
    ax.scatter(xCord2, yCord2, s=30, c='green')
    # plotting best fit (weights)
    # maxDelta = - x * delta(x) - y * delta(y)
    # weights[0] = - x * weights[1] - y * weights[2]
    x = arange(-3.0, 3.0, 0.1)
    y = (- weights[0] - weights[1] * x)/weights[2]
    ax.plot(x, y)
    # axis label
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

    
# classifier
# return 1 or 0 by calculating sigmoid with testX and weights
# sigmoid(0) = 0.5, if sigmoid > 0.5, guessing 1, else, 0
def classifier(testX, weights):
    probability = sigmoid(sum(testX * weights))
    if probability > 0.5:
        return 1.0
    else:
        return 0.0


#--------------------------------------------------------------
# testing


# testing
# return errorRate of testing after calculatation
def testing(dataTrain, dataTest):

    # dataLoad
    dataTrain = open(dataTrain)
    dataTest = open(dataTest)

    # (trainAttributes, trainCategories) spliting training data
    trainAttributes = []
    trainCategories = []   
    for line in dataTrain.readlines():
        thisLine = line.strip().split('\t')
        thisLineAttributes = []
        for i in range(21):
            thisLineAttributes.append(float(thisLine[i]))
        trainAttributes.append(thisLineAttributes)
        trainCategories.append(float(thisLine[21]))

    # (trainWeights) get weights by random gradient ascent from training data
    trainWeights = modifiedStocGradientAscent(trainAttributes, trainCategories, 500)


    # comparing categories of test data by classifier
    error = 0
    rowsTest = 0.0
    for line in dataTest.readlines():
        rowsTest += 1.0
        thisLine = line.strip().split('\t')
        thisLineAttributes = []
        for i in range(21):
            thisLineAttributes.append(float(thisLine[i]))
        if int(classifier(array(thisLineAttributes), trainWeights)) != int(thisLine[21]):
            error += 1
    errorRate = float(error) / rowsTest
    print "errorRate: %f" % errorRate
    return errorRate


# multiTesting
# iterating testing() for n times to evaluate
def multiTesting():
    error = 0.0
    multiTests = 10
    for i in range(multiTests):
        error += testing(dataTrain, dataTest)
    print "after %d iterations, the average error rate: %f" % (multiTests, error/float(multiTests))

dataTrain = "G:\\vimFiles\\python\\examples\\machine_learning_in_action\\horseTrain.txt"
dataTest = "G:\\vimFiles\\python\\examples\\machine_learning_in_action\\horseTest.txt"
testing(dataTrain, dataTest)



#--------------------------------------------------------------
# testing

dataPath = "G:\\vimFiles\\python\\examples\\machine_learning_in_action\\dataset.txt"
data, category = dataLoad(dataPath)

weights = gradientAscent(data, category)
print weights 
#bestFitPlot(weights, dataPath)

stochasticWeights = stochasticGradientAscent(data, category)
print stochasticWeights
stochasticWeights = mat(stochasticWeights).transpose()
#bestFitPlot(stochasticWeights, dataPath)

modifiedStocWeights = modifiedStocGradientAscent(data, category, steps=150)
print modifiedStocWeights
modifiedStocWeights = mat(modifiedStocWeights).transpose()
#bestFitPlot(modifiedStocWeights, dataPath)
