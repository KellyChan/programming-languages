'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Regression - linear regression
# Author: Kelly Chan
# Date: Jan 25 2014
------------------------------------------------------
'''




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
