'''
Created on Dec 30 2013
@author: Kelly Chan

Python Version: V2.7.3

Course: Python Data Mining
Lesson: Bayes
Methods: 

'''

class Bayes:
    def __init__(self, data):
        # here I am assuming the first column of the data is the class.
        self.data = data
        self.prior = {}
        self.conditional = {}
    
    def train(self):
        """train the Bayes Classifier
        basically a lot of counting"""
        total = 0
        classes = {}
        counts = {}
        # determine size of a training vector
        size = len(self.data[0])
        #
        # iterate through training instances
        for instance in self.data:
            total += 1
            category = instance[0]
            classes.setdefault(category, 0)
            counts.setdefault(category, {})
            classes[category] += 1
        # now process each column in instance
        col = 0
        for columnValue in instance[1:]:
            col += 1
            tmp = {}
            if col in counts[category]:
                tmp = counts[category][col]
                if columnValue in tmp:
                    tmp[columnValue] += 1
                else:
                    tmp[columnValue] = 1
                counts[category][col] = tmp
        # ok. done counting. now compute probabilities
        #
        # first prior probabilities
        #
        for (category, count) in classes.items():
            self.prior[category] = count / total
        # now compute conditional probabilities
        for (category, columns) in counts.items():
            tmp = {}
            for (col, valueCounts) in columns.items():
                tmp2 = {}
                for (value, count) in valueCounts.items():
                    tmp2[value] = count / classes[category]
                tmp[col] = tmp2
            #convert tmp to vector
            tmp3 = []
            for i in range(1, size):
                tmp3.append(tmp[i])
            self.conditional[category] = tmp3
    
    
    def classify(self, instance):
        categories = {}
        for (category, vector) in self.conditional.items():
            prob = 1
            for i in range(len(vector)):
                colProbability = .0000001
                if instance[i] in vector[i]:
                    # get the probability for that column value
                    colProbability = vector[i][instance[i]]
                prob = prob * colProbability
            prob = prob * self.prior[category]
            categories[category] = prob
        cat = list(categories.items())
        cat.sort(key=lambda catTuple: catTuple[1], reverse = True)
        return(cat[0])