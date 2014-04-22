#--------------------------------------------------------#
# Linear Structure: Queue
# Author: Kelly Chan
# Date: Apr 22 2014
#--------------------------------------------------------#

class Queue:

    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


class dequeue:

    def __init__(self):
        self.items = []

    def isEmpty(self):
         return self.items == []

     def addFront(self, item):
         self.items.append(item)

    def addRear(self, item):
        self.items.insert(0, item)

    def removeFront(self):
        return self.items.pop()

    def removeRear(self):
        return self.items.pop(0)

    def size(self):
        return len(self.items)


