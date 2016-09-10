'''
-------------------------------------------------------------------------
Book: Machine Learning In Action
# Lesson: MapReduce - reducer
# Author: Kelly Chan
# Date: Feb 3 2014
-------------------------------------------------------------------------
'''

import sys
from numpy import mat, mean, power

def dataLoad(dataFile):
    for line in dataFile:
        yield line.rstrip()


# creating a list of lines from dataFile
data = dataLoad(sys.stdin)
       
# spliting data lines into separte items and storing in list of lists
mapperOut = [line.split('\t') for line in data]

# accumulating total number of samples, overall sum and overall sum squared
accumulateN = 0.0
accumulateSum = 0.0
accumulateSumSquared = 0.0

for instance in mapperOut:
    thisN = float(instance[0])
    accumulateN += thisN
    accumulateSum += thisN * float(instance[1])
    accumulateSumSquared += thisN * float(instance[2])

# calculating means
mean = accumulateSum / accumulateN
meanSq = accumulateSumSquared / accumulateN

# printing size, mean, mean squared
print "%d\t%f\t%f" % (accumulateN, mean, meanSq)
print >> sys.stderr, "report: still alive"
