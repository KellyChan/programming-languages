'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Classification - decision tree
# Author: Kelly Chan
# Date: Jan 21 2014
------------------------------------------------------
'''


from math import log
import operator

# shannonEntropy
# return entropy of a specific feature (single) in a dataSet
def shannonEntropy(dataSet, featureIndex):

    # (category: #) counting # of the categories
    featureLabels = {}
    for line in dataSet:
        featureLabel = line[featureIndex]
        if featureLabel not in featureLabels.keys():
            featureLabels[featureLabel] = 0
        featureLabels[featureLabel] += 1

    # (entropy) calculating entropy in the categories
    entropy = 0.0
    n = len(dataSet)
    for key in featureLabels:
        probability = float(featureLabels[key]) / n
        entropy -= probability * log(probability, 2)
    return entropy


# dataReduction
# return dataReduced by a specific feature value (specific feature EXCLUSIVE)
def dataReduction(dataSet, featureIndex, featureValue):
    dataReduced = []
    for line in dataSet:
        if line[featureIndex] == featureValue:
            dataSplit = line[:featureIndex]  # data before featureIndex
            dataSplit.extend(line[featureIndex+1:]) # data (before index + after index)
            dataReduced.append(dataSplit)
    return dataReduced



# featureChoice
# return the bestFeatureIndex by calculating entropy for each feature/col
def featureChoice(dataSet, categoryIndex):

    bestInfoGain = 0.0
    bestFeatureIndex = categoryIndex
    categoryEntropy = shannonEntropy(dataSet, categoryIndex)

    cols = len(dataSet[0]) - 1 # dataSet[0]: the first row, get # of cols
    for col in range(cols):
        featureEntropy = 0.0
        feature = [row[col] for row in dataSet]  # col vector
        featureValues = set(feature) # get dict of values
        for featureValue in featureValues:
            dataReduced = dataReduction(dataSet, col, featureValue) # col EXCLUSIVE
            probability = len(dataReduced) / float(len(dataSet))
            featureEntropy += probability * shannonEntropy(dataReduced, categoryIndex)
        featureInfoGain = categoryEntropy - featureEntropy 
        # featureEntropy smaller, featureInfoGain bigger
        if (featureInfoGain > bestInfoGain):
            bestInfoGain = featureInfoGain
            bestFeatureIndex = col
    return bestFeatureIndex


# majorityLabel
# return majorityLabel in a feature/category vector
def majorityLabel(feature):
    labels = {}
    for label in feature:
        if label not in labels.keys():
            labels[label] = 0
        labels[label] += 1
    labels = sorted(labels.iteritems(), key=operator.itemgetter(1), reverse=True)
    return labels[0][0]


# tree
# return tree by deducting the best feature one by one (recursion)
def tree(dataSet, featureNames, categoryIndex):

    # get the categories vector
    categories = [row[categoryIndex] for row in dataSet]
    # if just one category in the categories
    if categories.count(categories[0]) == len(categories):
        return categories[0]
    # if just one column, return the majorityLabel in the col labels
    if len(dataSet[0]) == 1:
        return majorityLabel(categories)

    # (bestFeatureIndex, bestFeatureName, featureNames) constructing bestFeature 
    bestFeatureIndex = featureChoice(dataSet, categoryIndex)
    bestFeatureName = featureNames[bestFeatureIndex]
    del(featureNames[bestFeatureIndex]) # delete bestFeatureName from featureNames
    
    # (bestFeature, bestFeatureValues)
    bestFeature = [row[bestFeatureIndex] for row in dataSet]
    bestFeatureValues = set(bestFeature)

    # (treeOutput) constructing tree by recursion
    treeOutput = {bestFeatureName: {}}
    for bestFeatureValue in bestFeatureValues:
        subFeatureNames = featureNames[:]
        treeOutput[bestFeatureName][bestFeatureValue] = tree(dataReduction(dataSet, bestFeatureIndex, bestFeatureValue), 
                                                             subFeatureNames, 
                                                             categoryIndex)
    return treeOutput




dataSet = [[1, 1, 'yes'],
           [1, 1, 'yes'],
           [1, 0, 'no'],
           [0, 1, 'no'],
           [0, 1, 'no']]
featureNames = ['no surfacing','flippers']

#print shannonEntropy(dataSet, -1)
#print dataReduction(dataSet, 1,1)
#print featureChoice(dataSet, -1)
print tree(dataSet, featureNames, -1)
