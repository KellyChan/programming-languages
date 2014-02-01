'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Unsupervised Learning - FP growth
# Author: Kelly Chan
# Date: Jan 31 2014
------------------------------------------------------
'''

'''

Function Tree

- frequencedItemsList
     |
     |------------ mineTree
                      |
                      |------ findPrefixPath
                      |           |----------- ascendTree
                      |
                      |------ createTree
                                  |----------- (treeNode)
                                  |----------- udpateTree
                                                  |-------- (treeNode)
                                                  |-------- updateHeader

- mineTree
    |
    |------ createTree (fpTree, headerTable)
               |
               |-------- createInitSet
                              |---------- dataLoad

'''


# treeNode
# creating fpTree structure
# fpTree: 
# - nodeName - occurrence
#            - nodeLink    - parent
#                          - children
class treeNode:
    def __init__(self, nodeName, occurrence, parent):
        self.nodeName = nodeName
        self.occurrence = occurrence
        self.nodeLink = None
        self.parent = parent
        self.children = {}
    
    # counting the frequencies of occurrence
    def countFreq(self, occurrence):
        self.occurrence += occurrence

    # displaying the tree
    def display(self, tabs=1):
        print '  '*tabs, self.nodeName, ' ', self.occurrence
        # recursing the children nodes
        for child in self.children.values():
            child.display(tabs+1)

# createTree
# (fpTree, headerTable) 
# return fpTree and headerTable with items and freq(>=minFreq) from dataSet
# dataSet: extracting from rawData, must be [set]
def createTree(dataSet, minFreq=1):

    # generating headerTable from dataSet
    headerTable = {}
    # getting items and their frequencies from dataSet
    for transaction in dataSet:
        for item in transaction:
            headerTable[item] = headerTable.get(item, 0) + dataSet[transaction]
    # removing items those not meet the minFreq
    for key in headerTable.keys():
        if headerTable[key] < minFreq:
            del(headerTable[key])

    # checking the items/keys in headerTable
    itemsHeaderTable = set(headerTable.keys())
    #print 'itemsSet: ', itemsSet
    # if no items in headerTable, return None for fpTree and headerTable
    if len(itemsHeaderTable) == 0:
        return None, None

    # reformating headerTable to meet class treeNode
    # before reformating:
    # headerTable: {item: frequency, item: frequency}
    # after reformating:
    # headerTable: {item: [frequency, parent], item: [frequency, parent]}
    for key in headerTable:
        headerTable[key] = [headerTable[key], None]
    #print 'headerTable: ', headerTable

    # creating fpTree
    fpTree = treeNode('Null Set', 1, None)
    for transaction, frequency in dataSet.items():
        # creating transactionDict from headerTable
        transactionDict = {}
        for item in transaction:
            if item in itemsHeaderTable:
                # localDict: itemFreq = itemFreq in headerTable
                # headerTable[item][0]: frequency, headerTable: {item: [frequency, parent], item: [frequency, parent]}
                transactionDict[item] = headerTable[item][0]
        if len(transactionDict) > 0:
            # transactionDict.items(): get (item, frequency) from transactionDict
            # sorted(): sorting the items with frequnecy by descending order
            # orderedItems: get items by frequency with descending order
            orderedItems = [item[0] for item in sorted(transactionDict.items(), key=lambda freq: freq[1], reverse=True)]
            # populating tree with ordered frequency items
            updateTree(orderedItems, fpTree, headerTable, frequency)
    return fpTree, headerTable


# updateTree
# most freq item -> fpTree.children -> headerTable parent
# - updating fpTree.children with the most freq item
# - updating headerTable parent with fpTree.children
def updateTree(orderedItems, fpTree, headerTable, frequency):

    # orderedItems[0]: the most freq item
    # - as fpTree.children: 
    #      - if existing, add frequency
    #      - else, create fpTree.children with this item by treeNode
    #              update headerTable parent <- new fpTree.children    
    if orderedItems[0] in fpTree.children:
        # if fpTree.children is existing, update frequency
        fpTree.children[orderedItems[0]].countFreq(frequency)
    else:
        # if fpTree.children is NOT existing, create fpTree.children 
        fpTree.children[orderedItems[0]] = treeNode(orderedItems[0], frequency, fpTree)
        # update headerTable parent with fpTree.children
        if headerTable[orderedItems[0]][1] == None:
            # if no this parent, headerTable parent <- fpTree.chidren
            headerTable[orderedItems[0]][1] = fpTree.children[orderedItems[0]]
        else:
            # if parent, calling updateHeader()
            updateHeader(headerTable[orderedItems[0]][1], fpTree.children[orderedItems[0]])

    # if orderedItems is more than 1, calling updateTree to update the remaining ordered items 
    if len(orderedItems) > 1:
        updateTree(orderedItems[1::], fpTree.children[orderedItems[0]], headerTable, frequency)

# updateHeader
# updating parentHeaderTable <- fpTree.children 
def updateHeader(parentHeaderTable, childrenFPTree):
    while (parentHeaderTable.nodeLink != None):
        parentHeaderTable = parentHeaderTable.nodeLink
    parentHeaderTable.nodeLink = childrenFPTree


# ascendTree
# ascending from leaf node to root
def ascendTree(nodeHeaderTable, prefixPath):
    if nodeHeaderTable.parent != None:
        # appending the nodeName of nodeHeaderTable to prefixPath
        prefixPath.append(nodeHeaderTable.nodeName)
        # recursing the parent of nodeHeaderTable
        ascendTree(nodeHeaderTable.parent, prefixPath)

# findPrefixPath
# (conditionPaths) return conditionPaths by looping nodeHeaderTable's nodeLink
# - searching the ascending tree
# - return the occurrence
def findPrefixPath(item, nodeHeaderTable):
    conditionPaths = {}
    while nodeHeaderTable != None:
        prefixPath = []
        ascendTree(nodeHeaderTable, prefixPath)
        if len(prefixPath) > 1:
            # item -> prefixPath[0]
            # prefixPath[1:]: excluding item/prefixPath[0]
            conditionPaths[frozenset(prefixPath[1:])] = nodeHeaderTable.occurrence
        nodeHeaderTable = nodeHeaderTable.nodeLink
    return conditionPaths
    

# mineTree
# appending frequencedItemsList by prefix-conditionPaths
def mineTree(fpTree, headerTable, minFreq, prefixSet, frequencedItemsList):

    # get the items in headerTable by frequency with ascending order
    # headerTable.items: (item, frequency)
    # item: item[1] - frequency
    ascItems = [item[0] for item in sorted(headerTable.items(), key=lambda item: item[1])]

    # mining the items in headerTable one by one (ASCENDING order)
    # ascending order: the bottom of fpTree
    for item in ascItems:

        # adding items into prefixSet one by one
        thisPrefixSet = prefixSet.copy()
        thisPrefixSet.add(item)
        #print 'thisPrefixSet: ', thisPrefixSet
        frequencedItemsList.append(thisPrefixSet)



        # creating the conditionPaths
        conditionPaths = findPrefixPath(item, headerTable[item][1])
        #print 'conditionPaths: ', item, conditionPaths

        # creating conditionFPTree, coonditionHeaderTable by conditionPaths
        conditionFPTree, conditionHeaderTable = createTree(conditionPaths, minFreq)
        #print 'conditionHeaderTable from conditionFPTree: ', conditionHeaderTable

        # printing thisPrefixSet, conditionFPTree, mining the tree with thisPrefixSet
        if conditionHeaderTable != None:
            print 'conditionFPTree for: ', thisPrefixSet
            conditionFPTree.display(1)
            mineTree(conditionFPTree, conditionHeaderTable, minFreq, thisPrefixSet, frequencedItemsList)





#------------------------------------------------------
# testing

#rootNode = treeNode('pyramid',9, None)
#rootNode.children['eye'] = treeNode('eye', 13, None)
#rootNode.disp()

#rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
#rootNode.disp()


def dataLoad():
    rawData = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return rawData


def createInitSet(data):
    initSet = {}
    for transaction in data:
        initSet[frozenset(transaction)] = 1
    return initSet

testData = dataLoad()
#print "testData:\n",testData

initSet = createInitSet(testData)
#print "\ninitSet:\n",initSet

fpTree, headerTable = createTree(initSet, 3)
fpTree.display()

#print findPrefixPath('x', headerTable['x'][1])
#print findPrefixPath('z', headerTable['z'][1])
#print findPrefixPath('r', headerTable['r'][1])

# mining the frequencedItems
frequencedItems = []
mineTree(fpTree, headerTable, 3, set([]), frequencedItems)
print frequencedItems
