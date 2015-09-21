#--------------------------------------------------------#
# Linear Structure: Stack
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

def test():

    s=Stack()
    print(s.isEmpty())

    s.push(4)
    s.push('dog')
    print(s.peek())

    s.push(True)
    print(s.size())
    print(s.isEmpty())

    s.push(8.4)
    print(s.pop())
    print(s.pop())
    print(s.size())

if __name__ == '__main__':
    test()
