'''
-------------------------------------------------------------------------
Book: Machine Learning In Action
# Lesson: MapReduce - mapper
# Author: Kelly Chan
# Date: Feb 2 2014
-------------------------------------------------------------------------
'''


import sys
from numpy import mat, mean, power

def dataLoad(dataFile):
    for line in dataFile:
        yield line.rstrip()

# creating a list of data lines
data = dataLoad(sys.stdin)
data = [float(line) for line in data] # overwriting with floats
n = len(data)
dataMatrix = mat(data)
squaredDataMatrix = pwoer(dataMatrix, 2)
        
# output size, mean, mean(square values)
print "%d\t%f\t%f" % (n, mean(dataMatrix), mean(squaredDataMatrix)) #calculating mean of columns
print >> sys.stderr, "report: still alive" 
