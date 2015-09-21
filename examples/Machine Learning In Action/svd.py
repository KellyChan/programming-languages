'''
-------------------------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Dimensionality Reduction - SVD (singular value decomposition)
# Author: Kelly Chan
# Date: Feb 2 2014
-------------------------------------------------------------------------
'''


from numpy import *

#---------------------------------------------------------
# SVD

def dataLoad():
    return[[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]



U, Sigma, VT = linalg.svd([[1, 1],[7, 7]])
#print U
#print Sigma
#print VT


data = dataLoad()
U, Sigma, VT = linalg.svd(data)
#print Sigma

Sigma3 = mat([[Sigma[0], 0, 0],
              [0, Sigma[1], 0],
              [0, 0, Sigma[2]]])
#print Sigma3
#print U[:,:3] * Sigma3 * VT[:3,:]

#---------------------------------------------------------
# Recommendation System

'''

Function Tree

- recommend 
     |
     |------- dataLoad
     |
     |------- similarityMeans
     |                 |------- similarityEuclidean
     |                 |------- similarityPearson
     |                 |------- similarityCosine
     |
     |------- estMethod
                  |-------- standardEstimation
                  |-------- svdEstimation

'''

# similarityEuclidean
# euclidean = X - Y
def similarityEuclidean(vectorA, vectorB):
    return 1.0/(1.0 + linalg.norm(vectorA - vectorB))

# similarityPearson
# pearson = corrcoef
def similarityPearson(vectorA, vectorB):
    if len(vectorA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(vectorA, vectorB, rowvar = 0)[0][1]

# similarityCosine
# cosine = A * B / |A| * |B|
def similarityCosine(vectorA, vectorB):
    AB = float(vectorA.T * vectorB)
    absAB = linalg.norm(vectorA) * linalg.norm(vectorB)
    return 0.5 + 0.5 * (AB/absAB)


# standardEstimation
# (estimatedScore) return estimatedScore of unratedItems by other similar users
# estimatedScore = ratingSimilarityTotal / similarityTotal
# ratingSimilarity = userRating * similarity
def standardEstimation(dataMatrix, user, similarityMeans, itemIndex):

    cols = shape(dataMatrix)[1]
    similarityTotal = 0.0
    ratingSimilarityTotal = 0.0

    for col in range(cols):

        userRating = dataMatrix[user,col]
        if userRating == 0:
            continue
        
        # get the user rows with [item] rating and [col] rating
        userIndex = nonzero(logical_and(dataMatrix[:,itemIndex].A > 0, dataMatrix[:,col].A > 0))[0]
        if len(userIndex) == 0:
            simiarity = 0
        else:
            # calculating the similarity between item and other items 
            similarity = similarityMeans(dataMatrix[userIndex, itemIndex], dataMatrix[userIndex, col])
        print "itemIndex: %d, comparedItemIndex: %d, similarity: %f" % (itemIndex, col, similarity)

        # calculating similarityTotal by each similarity item score
        similarityTotal += similarity
        ratingSimilarityTotal += userRating * similarity
 
    if similarityTotal == 0:
        return 0
    else:
        return ratingSimilarityTotal / similarityTotal


# svdEstimation
# (unratedItem, estimatedScore)
# return (unratedItem, estimatedScore) by svdEstimaiton (dataMatrix reduction) 
def svdEstimation(dataMatrix, user, similarityMeans, itemIndex):

    cols = shape(dataMatrix)[1]
    similarityTotal = 0.0
    ratingSimilarityTotal = 0.0

    # calculating SVD of dataMatrix
    U, Sigma, VT = linalg.svd(dataMatrix)
    # diagonal matrix: 4 cols
    sigma4 = mat(eye(4) * Sigma[:4])
    # creating transformedItems
    transformItems = dataMatrix.T * U[:,:4] * sigma4.T

    for col in range(cols):
        userRating = dataMatrix[user, col]
        if userRating == 0 or col == itemIndex:
            continue
        similarity = similarityMeans(transformItems[itemIndex,:].T, transformItems[col,:].T)
        print "itemIndex: %d, comparedItemIndex: %d, similarity: %f" % (itemIndex, col, similarity)

        similarityTotal += similarity
        ratingSimilarityTotal += userRating * similarity

    if similarityTotal == 0:
        return 0
    else:
        return ratingSimilarityTotal / similarityTotal



# recommend
# (unratedItem, estimatedScore) 
# return (unratedItem, estimatedScore) by calculating similarityMeans 
def recommend(dataMatrix, user, N=3, similarityMeans=similarityCosine, estMethod = standardEstimation):

    # get the unratedItems
    unratedItems = nonzero(dataMatrix[user,:].A==0)[1]
    # if all items were rated, return a message
    if len(unratedItems) == 0:
        return "You rated everything."

    # calculating item scores of unratedItems by similarityMeans
    itemScores = []
    for unratedItem in unratedItems:
        estimatedScore = estMethod(dataMatrix, user, similarityMeans, unratedItem)
        itemScores.append((unratedItem, estimatedScore))
    return sorted(itemScores, key=lambda item: item[1], reverse=True)[:N]

#-------------------------------------------------
# testing

def dataLoad():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

dataMatrix = mat(dataLoad())

#print "\nEuclidean:"
distEuclidean = similarityEuclidean(dataMatrix[:,0],dataMatrix[:,4])
#print distEuclidean
distEuclidean = similarityEuclidean(dataMatrix[:,0],dataMatrix[:,0])
#print distEuclidean

#print "\nConsine:"
distCosine = similarityCosine(dataMatrix[:,0], dataMatrix[:,4])
#print distCosine
distCosine = similarityCosine(dataMatrix[:,0], dataMatrix[:,0])
#print distCosine

#print "\nPearson:"
distPearson = similarityPearson(dataMatrix[:,0], dataMatrix[:,4])
#print distPearson
distPearson = similarityPearson(dataMatrix[:,0], dataMatrix[:,0])
#print distPearson


dataMatrix[0,1] = dataMatrix[0,0] = dataMatrix[1,0] = dataMatrix[2,0] = 4
dataMatrix[3,3] = 2
#print dataMatrix
#print recommend(dataMatrix, 2)
#print recommend(dataMatrix, 2, similarityMeans=similarityEuclidean)
#print recommend(dataMatrix, 2, similarityMeans=similarityPearson)





def dataLoad():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

dataMatrix = mat(dataLoad())
U, Sigma, VT = linalg.svd(dataMatrix)
print "Sigma: \n", Sigma

squaredSigma = Sigma**2
print "total energy: ", sum(squaredSigma)
print "energy (90%): ", sum(squaredSigma) * 0.9
print "energy (first two elements): ", sum(squaredSigma[:2])
print "energy (first three elements): ", sum(squaredSigma[:3])
# to reduce 11 dimensions to 4 dimensions
print recommend(dataMatrix, 1, estMethod=svdEstimation)
print recommend(dataMatrix, 1, estMethod=svdEstimation, similarityMeans=similarityPearson)



#---------------------------------------------------------
# Image Compression


# imgPrint
# printing image
def imgPrint(dataMatrix, threshold=0.8):
    for row in range(32):
        for col in range(32):
            if float(dataMatrix[row, col]) > threshold:
                print 1,
            else:
                print 0,
        print ''


# imgCompress
# compressing image by SVD 
def imgCompress(imgFile, dims=3, threshold=0.8):
    
    # extracting img froom raw imgFile
    img = []
    for line in open(imgFile).readlines():
        thisRow = []
        for i in range(32):
            thisRow.append(int(line[i]))
        img.append(thisRow)

    # printing the original imgMatrix
    imgMatrix = mat(img)
    print "****original matrix******"
    imgPrint(imgMatrix, threshold)

    # compressing imgMatrix by SVD
    U, Sigma, VT = linalg.svd(imgMatrix)
    sigmaReducedMatrix = mat(zeros((dims, dims)))
    for dim in range(dims):
        sigmaReducedMatrix[dim, dim] = Sigma[dim]
    reconstructedMatrix = U[:,:dims] * sigmaReducedMatrix * VT[:dims,:]

    print "****reconstructed matrix using %d singular values******" % dims
    imgPrint(reconstructedMatrix, threshold)



imgFile = "G:\\vimFiles\\python\\examples\\machine_learning_in_action\\dataImg.txt"
imgCompress(imgFile)

