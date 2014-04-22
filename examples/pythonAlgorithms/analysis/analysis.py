#--------------------------------------------------------#
# Algorithms: Analysis
# Author: Kelly Chan
# Date: Apr 22 2014
#--------------------------------------------------------#

def sumOfN(n):
    theSum = 0
    for i in range(1, n+1):
        theSum += theSum
    return theSum

def foo(tom):
    fred = 0
    for bill in range(1, tom+1):
        barney = bill
        fred += barney
    return fred


def test():
    print(sumOfN(10))

if __name__ == '__main__':
    test()



