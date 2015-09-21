'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Dimensionality Reduction - PCA (principal component analysis)
# Author: Kelly Chan
# Date: Feb 1 2014
------------------------------------------------------
'''


from numpy import *


# dataLoad
# (dataMatrix) extracting dataMatrix from rawData
def dataLoad(dataFile, delim ='\t'):
    rawData = open(dataFile)
    features = [line.strip().split(delim) for line in rawData.readlines()]
    data = [map(float, line) for line in features]
    return mat(data)

# NaNtoMean
# replacing NaN <- Mean
def NaNtoMean():
    dataMatrix = dataLoad(dataFile, '\t')
    cols = shape(dataMatrix)[1]
    for col in range(cols):
        # meanFeatures: calculating means of each feature (NaN exclusive)
        meanFeatures = mean(dataMatrix[nonzero(~isnan(dataMatrix[:,col].A))[0],col])
        # NaN <- mean
        dataMatrix[nonzero(isnan(dataMatrix[:,col].A))[0],col] = meanFeatures
    return dataMatrix


# pca
# (reducedDataMatrix, reconstructedMatrix)
# reducedDataMatrix = meanRemoved * reducedEigenVectors
# reconstructedMatrix = (reducedDataMatrix * reducedEigenVectors.T) + meanFeatures
#
# reducedEigenVectors <- eigenVecotrs[:, reducedEigenValuesIndex] <- eigenValues
# eigenValues, eigenVectors <- covariance <- meanRemoved
def pca(dataMatrix, featuresTopN=9999999):

    # calculating mean by each col/feature
    meanFeatures = mean(dataMatrix, axis=0)
    meanRemovedMatrix = dataMatrix - meanFeatures

    # calculating covarianceMatrix with meanRemovedMatrix
    covarianceMatrix = cov(meanRemovedMatrix, rowvar=0)
    # calculating eigenValues, eigenVectors with covarianceMatrix
    eigenValues, eigenVectors = linalg.eig(mat(covarianceMatrix))

    # sorting: smallest -> largest
    # argsort(): get the order of the eigenvalues
    eigenValuesIndex = argsort(eigenValues)
    # cutting  off unwanted dimensions
    eigenValuesIndex = eigenValuesIndex[:-(featuresTopN+1):-1]

    # reorganizing eigenVectors: largest -> smallest
    reducedEigenVectors = eigenVectors[:, eigenValuesIndex]

    # transforming data into new dimensions
    reducedDataMatrix = meanRemovedMatrix * reducedEigenVectors
    reconstructedMatrix = (reducedDataMatrix * reducedEigenVectors.T) + meanFeatures
    return reducedDataMatrix, reconstructedMatrix



#----------------------------------------------------
# testing


dataFile = "G:\\vimFiles\\python\\examples\\machine_learning_in_action\\dataPCA.txt"
#dataMatrix = NaNtoMean()
dataMatrix = dataLoad(dataFile)
reducedDataMatrix, reconstructedMatrix = pca(dataMatrix,1)
print shape(reducedDataMatrix)



import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(dataMatrix[:,0].flatten().A[0], dataMatrix[:,1].flatten().A[0], marker='^', s=90)
#ax.scatter(reconstructedMatrix[:,0].flatten().A[0], reconstructedMatrix[:,1].flatten().A[0], marker='o', s=50, c='red')
#plt.show()
