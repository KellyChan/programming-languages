'''
Created on Dec 30 2013
@author: Kelly Chan

Python Version: V2.7.3

Course: Python Data Mining
Lesson: Classification
Methods: 

'''


def evalClassifier(self, filename):
    """evaluate test set data"""
    f = open(filename)
    total = 0
    correct = 0
    for line in f:
        elements = line.split('\t')
        if len(elements) >= 5:
            total += 1
            name = elements[0]
            sport = elements[1]
            # not using age which is elements[2]
            height = int(elements[3])
            weight = int(elements[4])
            classification = self.classify(name, (height, weight))
            if classification == sport:
                print("%s CORRECT" % name)
                correct += 1
            else:
                print("%s MISCLASSIFIED AS %s. Should be %s" % (name, classification, sport))
                print("%f correct" % (correct / total))