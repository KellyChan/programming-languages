'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Unsupervised Learning - apriori
# Author: Kelly Chan
# Date: Jan 30 2014
------------------------------------------------------
'''

'''

Function Tree

apriori: finding the frequenced items
rulesGen: generating rules from frequenced items (apriori)

- rulesGen
    |
    |------- apriori
    |           |-------- aprioriGen
    |           |-------- itemScan
    |                       |------- itemsExtracted
    |                                     |--------- dataLoad
    |
    |
    |------- rulesFromConsequence
    |                |
    |------- confidenceCalculation  


'''


from numpy import *

# dataLoad
# extracting data from rawData
def dataLoad():
    return [[1, 3, 4], 
            [2, 3, 5], 
            [1, 2, 3, 5], 
            [2, 5]]


# itemsExtracted
# (forzenItems) return individual items / rawElements from data
def itemsExtracted(data):
    items = []
    for transaction in data:
        for item in transaction:
            if not [item] in items:
                items.append([item])
    items.sort()
    # forzenset: use it as a key in a dict
    return map(frozenset, items)


# itemScan
# (itemSupported, itemProbabilityDict)
# itemSupported: return item if its probablity >= minProbability
# itemProbabilityDict: return probability of each item
def itemScan(transactionSets, frozenItems, minProbabilitySupport):

    # counting the frequency of each item
    itemFreq = {}
    for transaction in transactionSets:
        for item in frozenItems:
            if item.issubset(transaction):
                if not itemFreq.has_key(item):
                    itemFreq[item] = 1
                else:
                    itemFreq[item] += 1

    n = float(len(transactionSets))
    itemSupported = []
    itemProbabilityDict = {}
    for key in itemFreq:
        probability = itemFreq[key] / n
        itemProbabilityDict[key] = probability
        if probability >= minProbabilitySupport:
            itemSupported.insert(0,key)

    return itemSupported, itemProbabilityDict


# aprioriGen
# (itemCombinations) return itemCombinations after combining k subsets
# itemSupported: generated from itemScan
# k: how many subsets for this combination
def aprioriGen(itemSupported, k):
    itemCombinations = []
    sets = len(itemSupported)
    for i in range(sets):
        for j in range(i+1, sets):
            # convert frozenset of itemSupported to list
            # list(itemSupported[i]) each set in itemSupported
            # [: k - 2]: get the first item in this set 
            firstItemInSet1 = list(itemSupported[i])[: k - 2]
            firstItemInSet2 = list(itemSupported[j])[: k - 2]
            firstItemInSet1.sort()
            firstItemInSet2.sort()
            # if both items are the same, union and get one
            if firstItemInSet1 == firstItemInSet2:
                itemCombinations.append(itemSupported[i] | itemSupported[j])
    return itemCombinations


# apriori
# (itemSupportedList, itemProbabilityDict)
# return itemSupportedList, itemProbabilityDict by aprioriGen
def apriori(data, minProbabilitySupport = 0.5):
    frozenItems = itemsExtracted(data)
    transactionSets = map(set, data)
    itemSupported, itemProbabilityDict = itemScan(transactionSets, frozenItems, minProbabilitySupport)

    itemSupportedList = [itemSupported]
    k = 2
    while (len(itemSupportedList[k-2]) > 0):
        itemCombinations = aprioriGen(itemSupportedList[k-2], k)
        kItemSupported, kItemProbabilityDict = itemScan(transactionSets, itemCombinations, minProbabilitySupport)
        itemProbabilityDict.update(kItemProbabilityDict)
        itemSupportedList.append(kItemSupported)
        k += 1
    return itemSupportedList, itemProbabilityDict


# confidenceCalculation
# (prunedSubsetItems) return prunedSubsetItems by calculating confidence
def confidenceCalculation(subset, subsetItems, itemProbabilityDict, rules, minConfidence=0.7):
    prunedSubsetItems = []
    for consequence in subsetItems:
        confidence = itemProbabilityDict[subset] / itemProbabilityDict[subset-consequence]
        if confidence >= minConfidence:
            print subset-consequence, '-->', consequence,'confidence:',confidence
            rules.append((subset-consequence, consequence, confidence))
            prunedSubsetItems.append(consequence)
    return prunedSubsetItems



# rulesFromConsequence
# generating rules by itemCombinations in subset
def rulesFromConsequence(subset, subsetItems, itemProbabilityDict, rules, minConfidence=0.7):
    # get the first frozenset in subsetItems
    n = len(subsetItems[0])
    # if # of subset > # of first frozenset + 1
    if (len(subset) > (n+1)):
        itemCombinations = aprioriGen(subsetItems, n+1)
        itemCombinations = confidenceCalculation(subset, itemCombinations, itemProbabilityDict, rules, minConfidence)
        if (len(itemCombinations) > 1):
            rulesFromConsequence(subset, itemCombinations, itemProbabilityDict, rules, minConfidence)



# rulesGen
# (rules) return rules by rulesFromConsequence and confidenceCalculation
# itemSupportedList: gengerated by apriori
# itemProbabilityDict: generated by itemScan
def rulesGen(itemSupportedList, itemProbabilityDict, minConfidence=0.7):
    rules = []
    for i in range(1, len(itemSupportedList)): # only get the sets with two or more items
        for subset in itemSupportedList[i]:
            subsetItems = [frozenset([item]) for item in subset]
            if (i > 1):
                rulesFromConsequence(subset, subsetItems, itemProbabilityDict, rules, minConfidence)
            else:
                confidenceCalculation(subset, subsetItems, itemProbabilityDict, rules, minConfidence)
    return rules


#-----------------------------------------------------
# testing

data = dataLoad()
print data
frozenItems = itemsExtracted(data)
print "\nFrozenItems:\n", frozenItems

transactionSets = map(set, data)
print "\ntransactionSets:\n", transactionSets

supportedItems, supportDict = itemScan(transactionSets, frozenItems, 0.5)
print "\nsupportedItems\n", supportedItems
print "\nsupportDict\n", supportDict

supportedItems, supportedDict = apriori(data, minProbabilitySupport = 0.5)
print "\nsupportedItems:\n",supportedItems

print "\nselection\n", supportedItems[0]
print "\ngen:\n",aprioriGen(supportedItems[0],2)

rules = rulesGen(supportedItems, supportedDict, minConfidence=0.5)
print "\nrules:\n", rules
