#--------------------------------------------------------#
# recurion
# Author: Kelly Chan
# Date: Apr 22 2014
#--------------------------------------------------------#

class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items)-1]

    def size(self):
        return len(self.items)



def listsum(numList):
    theSum = 0
    for i in numList:
        theSum += i
    return theSum

def listsum(numList):
    if len(numList) == 1:
        return numList[0]
    else:
        return numList[0] + listsum(numList[1:])


def toStr(n, base):
    convertString = "0123456789ABCDEF"
    if n < base:
        return convertString[n]
    else:
        return toStr(n//base, base) + convertString[n%base]


rStack = Stack()
def toStr(n, base):

    convertString = "0123456789ABCDEF"
    while n > 0:
        if n < base:
            rStack.push(convertString[n])
        else:
            rStack.push(convertString[n % base])
        n = n // base

    res = ""
    while not rStack.isEmpty():
        res = res + str(rStack.pop())

    return res



def test():

    print(listsum([1,3,5,7,9]))
    print(toStr(1453,16))


if __name__ == "__main__":
    test()

