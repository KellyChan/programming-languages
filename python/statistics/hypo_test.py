#Complete the test function to perform a hypothesis test 
#on list l under the null that the mean is h
 
from math import sqrt
 
def mean(l):
    return float(sum(l))/len(l)
 
def var(l):
    m = mean(l)
    return sum([(x-m)**2 for x in l])/len(l)
 
def factor(l):
    return 1.96
 
 
def conf(l):
    return factor(l) * sqrt(var(l) / len(l))
 
 
def test(l, h):
    m = mean(l)
    c = conf(l)
    return abs(h-m) <= c