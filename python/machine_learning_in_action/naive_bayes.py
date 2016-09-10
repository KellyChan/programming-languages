'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Classification - naive bayes
# Author: Kelly Chan
# Date: Jan 22 2014
------------------------------------------------------
'''


'''
Function Tree

- testing
    |
- classifier
    |
- naiveBayes
       |----- checkDataInWordsDict
                    |------------- vocabularyDict
                    |-------------------|---------- dataLoad
'''

from numpy import *

# dataLoad
# (textList, categories) return textList, categories from raw data
def dataLoad():

    categories = [0,1,0,1,0,1] #1 is abusive, 0 not
    textList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
              ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
              ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
              ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
              ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
              ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    return textList, categories


# vocabularyDict
# (wordsDict) return words dictionary from dataset
def vocabularyDict(dataset):
    wordsDict = set([])
    for document in dataset:
        wordsDict = wordsDict | set(document)
    return list(wordsDict)


# checkDataInWordsDict
# (checkVector) return checkVector if word is in wordList but not in worksDict 
def checkDataInWordsDict(wordsDict, wordList):
    checkWords = [0]*len(wordsDict)
    for word in wordList:
        if word in wordsDict:
            checkWords[wordsDict.index(word)] = 1
        else:
            print "this word is not in the vocabulary: %s" % word
    return checkWords


# naiveBayes
# (checkWordsMatrix, categories) 
# return probabilityTrue, probabilityFalse, probabilityCategory by calculating bayes
def naiveBayes(checkWordsMatrix, categories):

    rows = len(checkWordsMatrix)
    cols = len(checkWordsMatrix[0])
    vectorTrue = zeros(cols)
    vectorFalse = zeros(cols)

    totalTrue = 0.0
    totalFalse = 0.0
    for i in range(rows):
        if categories[i] == 1:
            vectorTrue += checkWordsMatrix[i]
            totalTrue += sum(checkWordsMatrix[i])
        else:
            vectorFalse += checkWordsMatrix[i]
            totalFalse += sum(checkWordsMatrix[i])

    probabilityCategory = sum(categories) / float(rows)
    vectorProbabilityTrue = vectorTrue / totalTrue # change to log
    vectorProbabilityFalse = vectorFalse / totalFalse # change to log
    return probabilityCategory, vectorProbabilityTrue, vectorProbabilityFalse


# classifier
# return category by comparing probability
# ln f(x) ~ f(x): 
# whatever x, trends increased and decreased are the same for lnf(x) and f(x).
# ln(a*b) = lna + lnb: 
# Since probabilities in vector are very small, they become bigger after log.
def classifier(checkWordsTestX, vectorProbabilityTrue, vectorProbabilityFalse, probabilityCategory):
    probabilityTrue = sum(checkWordsTestX * vectorProbabilityTrue) + log(probabilityCategory)
    probabilityFalse = sum(checkWordsTestX * vectorProbabilityFalse) + log(1.0 - probabilityCategory)
    if probabilityTrue > probabilityFalse:
        return 1
    else:
        return 0


# testing
# return category of test data by classifier
def testing():

    # training
    #
    # (data, category) dataLoad
    data, category = dataLoad()
    # (wordsDict) vocabularyDict
    wordsDict = vocabularyDict(data)
    # (checkWordsMatrix) naiveBayes Training
    checkWordsMatrix = []
    for line in data:
        checkWordsMatrix.append(checkDataInWordsDict(wordsDict, line))
    probabilityCategory, vectorProbabilityTrue, vectorProbabilityFalse = naiveBayes(checkWordsMatrix, category)


    # testing
    #
    # dataLoad
    testEntry = [['love', 'my', 'dalmation'],
                 ['stupid', 'garbage']]
    # (checkWordsTestX) checkDataInWordsDict, classifier
    for i in range(len(testEntry)):
        checkWordsTestX = array(checkDataInWordsDict(wordsDict, testEntry[i]))
        print testEntry[i],'classified as: ',classifier(checkWordsTestX, vectorProbabilityTrue, vectorProbabilityFalse, probabilityCategory)



testing()


#--------------------------------------------------------------
# Testing

from numpy import *

dataset, abusive = dataLoad()
wordsDict = vocabularyDict(dataset)
#print wordsDict
checkWords = checkDataInWordsDict(wordsDict, dataset[3])

# converting checkWordsMatrix from raw data to wordsDict by lines
checkWordsMatrix = []
for line in dataset:
    checkWordsMatrix.append(checkDataInWordsDict(wordsDict, line))
vectorPobabilityTrue, vectorPobabilityFalse, probabilityAbusive = naiveBayes(checkWordsMatrix, abusive)
#print probabilityAbusive
#print vectorPobabilityTrue
#print vectorPobabilityFalse

