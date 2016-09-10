'''
Created on Dec 30 2013
@author: Kelly Chan

Python Version: V2.7.3

Course: Python Data Mining
Lesson: Classification
Methods: 

'''

music = {"Dr Dog/Fate": {"piano": 2.5, 
                         "vocals": 4, 
                         "beat": 3.5, 
                         "blues": 3,
                         "guitar": 5, 
                         "backup vocals": 4, 
                         "rap": 1},
         "Phoenix/Lisztomania": {"piano": 2,
                                "vocals": 5, 
                                "beat": 5, 
                                "blues": 3,
                                "guitar": 2, 
                                "backup vocals": 1, 
                                "rap": 1}}


users = {"Angelica": {"Dr Dog/Fate": "L", "Phoenix/Lisztomania": "L",
                      "Heartless Bastards/Out at Sea": "D",
                      "Todd Snider/Don't Tempt Me": "D", "The Black Keys/Magic Potion": "D",
                      "Glee Cast/Jessie's Girl": "L", "La Roux/Bulletproof": "D",
                      "Mike Posner": "D", "Black Eyed Peas/Rock That Body": "D",
                      "Lady Gaga/Alejandro": "L"},
         "Bill": {"Dr Dog/Fate": "L", "Phoenix/Lisztomania": "L",
                  "Heartless Bastards/Out at Sea": "L", "Todd Snider/Don't Tempt Me": "D",
                  "The Black Keys/Magic Potion": "L", "Glee Cast/Jessie's Girl": "D",
                  "La Roux/Bulletproof": "D", "Mike Posner": "D",
                  "Black Eyed Peas/Rock That Body": "D", "Lady Gaga/Alejandro":
                  "D"} }

         
#
# the item vector represents the attributes: piano, vocals, beat, blues,
#                                            guitar, backup vocals, rap
#
items = {"Dr Dog/Fate": [2.5, 4, 3.5, 3, 5, 4, 1],
         "Phoenix/Lisztomania": [2, 5, 5, 3, 2, 1, 1],
         "Heartless Bastards/Out at Sea": [1, 5, 4, 2, 4, 1, 1],
         "Todd Snider/Don't Tempt Me": [4, 5, 4, 4, 1, 5, 1],
         "The Black Keys/Magic Potion": [1, 4, 5, 3.5, 5, 1, 1],
         "Glee Cast/Jessie's Girl": [1, 5, 3.5, 3, 4, 5, 1],
         "La Roux/Bulletproof": [5, 5, 4, 2, 1, 1, 1],
         "Mike Posner": [2.5, 4, 4, 1, 1, 1, 1],
         "Black Eyed Peas/Rock That Body": [2, 5, 5, 1, 2, 2, 4],
         "Lady Gaga/Alejandro": [1, 5, 3, 2, 1, 2, 1]}

def manhattan(vector1, vector2):
    """Computes the Manhattan distance."""
    distance = 0
    total = 0
    n = len(vector1)
    for i in range(n):
        distance += abs(vector1[i] - vector2[i])
    return distance


def computeNearestNeighbor(itemName, itemVector, items):
    """creates a sorted list of items based on their distance to item"""
    distances = []
    for otherItem in items:
        if otherItem != itemName:
            distance = manhattan(itemVector, items[otherItem])
            distances.append((distance, otherItem))
    # sort based on distance -- closest first
    distances.sort()
    return distances


def classify(user, itemName, itemVector):
    """Classify the itemName based on user ratings
    Should really have items and users as parameters"""
    # first find nearest neighbor
    nearest = computeNearestNeighbor(itemName, itemVector, items)[0][1]
    rating = users[user][nearest]
    return rating






print classify('Angelica', 'Enya/Wild Child', [1, 5, 2.5, 1, 1, 5, 1])
print computeNearestNeighbor('Enya/Wild Child', [1, 5, 2.5, 1, 1, 5, 1], items)