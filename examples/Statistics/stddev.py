# the stddev function to make it return the standard deviation of a list of numbers
 
from math import sqrt
 
def mean(data):
    return sum(data)/len(data)
    
def variance(data):
    mu=mean(data)
    return mean([(x-mu)**2 for x in data])
    
def stddev(data):
    return sqrt(variance(data))