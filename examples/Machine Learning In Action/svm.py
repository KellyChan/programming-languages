'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Classification - support vector machine
# Author: Kelly Chan
# Date: Jan 24 2014
------------------------------------------------------
'''

from numpy import *

def dataLoad(dataPath):
    data = []
    categories = []
    rawData = open(dataPath)
    for line in rawData.readlines():
        thisLineAttributes = line.strip().split('        ')
        data.append([float(thisLineAttributes[0]), float(thisLineAttributes[1])])
        categories.append(float(thisLineAttributes[2]))
    return data, categories


def alphaIndexRandom(alphaIndex, nAlpha):
    alphaRandomIndex = nAlpha
    while (alphaRandomIndex == nAlpha):
        alphaRandomIndex = int(random.uniform(0,nAlpha))
    return alphaRandomIndex

def alphaClip(alphaRandom, H, L):
    if alphaRandom > H:
        alphaRandom = H
    if L > alphaRandom:
        alphaRandom = L
    return alphaRandom


def simplifiedSMO(data, categories, C, tolerance, maxLoop):
    dataMatrix = mat(data)
    categoryMatrix = mat(categories)
    b = 0
    rows, cols = shape(dataMatrix)
    alphaMatrix = mat(zeros((rows,1)))
    loop = 0

    while (loop < maxLoop):
        alphaPairsChanged = 0
        for i in range(rows):
            # W^T * X + b 
            f = float(multiply(alphaMatrix, categoryMatrix).T * (dataMatrix * dataMatrix[i,:].T)) + b
            e = f - float(categoryMatrix[i])
            if ((categoryMatrix[i] * e < -tolerance) and (alphaMatrix[i] < C)) or ((categoryMatrix[i] * e > tolerance) and (alphaMatrix[i] > 0)):
                    alphaRandomIndex = alphaIndexRandom(i, rows)
                    fAlpha = float(multiply(alphaMatrix, categoryMatrix).T * (dataMatrix * dataMatrix[alphaRandomIndex,:].T)) + b
                    eAlpha = fAlpha - float(categoryMatrix[alphaRandomIndex])
                    thisAlpha = alphaMatrix[i].copy()
                    randomAlpha = alphaMatrix[alphaRandomIndex].copy()
                    if (categoryMatrix[i] != categoryMatrix[alphaRandomIndex]):
                        L = max(0, alphaMatrix[alphaRandomIndex] - alphaMatrix[i])
                        H = min(C, C + alphaMatrix[alphaRandomIndex] - alphaMatrix[i])
                    else:
                        L = max(0, alphaMatrix[alphaRandomIndex] + alphaMatrix[i] - C)
                        H = min(C, alphaMatrix[alphaRandomIndex] + alphaMatrix[i])
                    if L == H:
                        print "L == H"
                        continue
                    eta = 2.0 * dataMatrix[i,:] * dataMatrix[alphaRandomIndex,:].T - \
                                dataMatrix[i, :] * dataMatrix[i, :].T - \
                                dataMatrix[alphaRandomIndex, :] * dataMatrix[alphaRandomIndex, :].T
                    if eta >= 0:
                        print "eta >= 0"
                        continue
                    alphaMatrix[alphaRandomIndex] -= categoryMatrix[alphaRandomIndex] * (e - eAlpha) / eta
                    alphaMatrix[alphaRandomIndex] = alphaClip(alphaMatrix[alphaRandomIndex], H, L)
                    if (abs(alphaMatrix[alphaRandomIndex] - thisAlpha) < 0.00001):
                        print "alphaRandomIndex is not moving enough"
                        continue
                    alphaMatrix[i] += categoryMatrix[alphaRandomIndex] * categoryMatrix[i] * (thisAlpha - alphaMatrix[alphaRandomIndex])
                    b1 = b - e - categoryMatrix[i] * (alphaMatrix[i] - thisAlpha) * dataMatrix[i, :] * dataMatrix[i, :].T \
                               - categoryMatrix[alphaRandomIndex] * \
                                 (alphaMatrix[alphaRandomIndex] - thisAlpha) * \
                                 dataMatrix[i, :] * dataMatrix[alphaRandomIndex, :].T
                    b2 = b - eAlpha - categoryMatrix[i] * (alphaMatrix[i] - thisAlpha) * dataMatrix[i, :] * dataMatrix[alphaRandomIndex, :].T - \
                                      categoryMatrix[alphaRandomIndex] * \
                                      (alphaMatrix[alphaRandomIndex] - thisAlpha) * \
                                      dataMatrix[alphaRandomIndex, :] * \
                                      dataMatrix[alphaRandomIndex, :].T
                    if (0 < alphaMatrix[i]) and (C > alphaMatrix[i]):
                        b = b1
                    elif (0 < alphaMatrix[alphaRandomIndex]) and (C > alphaMatrix[alphaRandomIndex]):
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    alphaPairsChanged += 1
                    print "loops: %d  i: %d, pairs changed %d" % (loop, i, alphaPairsChanged)
        if (alphaPairsChanged == 0):
            loop += 1
        else:
            loop = 0
        print "loop: %d" % loop

    return b, alphaMatrix




dataPath = "G:\\vimFiles\\python\\examples\\machine_learning_in_action\\datasetSVM.txt"
data, categories = dataLoad(dataPath)
print categories
#print simplifiedSMO(data, categories, 0.6, 0.001, 1)


