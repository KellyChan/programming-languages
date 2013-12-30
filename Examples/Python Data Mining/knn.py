'''
Created on Dec 30 2013
@author: Kelly Chan

Python Version: V2.7.3

Course: Python Data Mining
Lesson: Classification
Methods: KNN

'''

def kNN(self, itemName, itemVector, k):
    # first find nearest neighbor
    tmp = {}
    resultSet = self.computeNearestNeighbor(itemName,self.normalizeInstance(itemVector))[:k]
    # the resultSet now contains the k nearest neighbors
    for res in resultSet:
        classification = self.category[res[1]]
        if classification in tmp:
            tmp[classification] += 1
        else:
            tmp[classification] = 1
    recommendations = list(tmp.items())
    # recommendations is a list of classes and how many times
    # that class appeared in the nearest neighbor list (votes)
    # i.e. [['gymnastics', 2], ['track', 1]]
    recommendations.sort(key=lambda artistTuple: artistTuple[1], reverse = True)
    # construct list of classes that have the largest number of votes
    topRecommendations = list (filter(lambda k: k[1] == recommendations[0][1], recommendations))
    # if only one class has the highest number of votes return that class
    if len(topRecommendations) == 1:
        rating = topRecommendations[0][0]
    else:
        rint = random.randint(0, len(topRecommendations) - 1)
        rating = topRecommendations[rint][0]
    return rating