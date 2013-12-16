# CORRECT SPECIFICATION:
#
# the Queue class provides a fized-size FIFO queue of integers
#
# the constructor takes a single parameter: an integer >0 that
# is the maximum number of elements the queue can hold
#
# empty() returns True iff the queue holds no elements
#
# full() returns True iff the queue cannot hold any more elements
#
# enqueue(i) attempts to put the integer i into the queue; it returns
# True if successful and False if the queue is full
#
# dequeue() removes an integer from the queue and returns it,
# or else returns None if the queue is empty

import array

class Queue:
    def __init__(self,size_max):
        assert size_max > 0
        self.max = size_max
        self.head = 0
        self.tail = 0
        self.size = 0
        self.data = array.array('i', range(size_max))

    def empty(self):
        return self.size == 0

    def full(self):
        return self.size == self.max

    def enqueue(self,x):
        if self.size == self.max:
            return False
        self.data[self.tail] = x
        self.size += 1
        self.tail += 1
        if self.tail == self.max:
            self.tail = 0
        return True

    def dequeue(self):
        if self.size == 0:
            return None
        x = self.data[self.head]
        self.size -= 1
        self.head += 1
        if self.head == self.max:
            self.head = 0
        return x


def test1():
    q = Queue(3)
    res = q.empty()
    if not res:
        print "test1 NOT OK"
        return
    res = q.enqueue(10)
    if not res:
        print "test1 NOT OK"
        return
    res = q.enqueue(11)
    if not res:
        print "test1 NOT OK"
        return
    x = q.dequeue()
    if x != 10:
        print "test1 NOT OK"
        return
    x = q.dequeue()
    if x != 11:
        print "test1 NOT OK"
        return
    res = q.empty()
    if not res:
        print "test1 NOT OK"
        return
    print "test1 OK"

def test2():
    ###Your code here.
    q = Queue(2)
    res = q.empty()
    if not res:
        print "test2 NOT OK"
        return
    res = q.enqueue(1)
    if not res:
        print "test2 NOT OK"
        return
    res = q.enqueue(2)
    if not res:
        print "test2 NOT OK"
        return
    x = q.dequeue(3)
    if q.tail != 0:
        print "test2 NOT OK"
        return
    print "test2 OK"

def test3():
    ###Your code here.
    q = Queue(1)
    res = q.empty()
    if not res:
        print "test3 NOT OK"
        return
    x = q.dequeue()
    if not x is None:
        print "test3 NOT OK"
        return
    res = q.dequeue(3)
    if not res:
        print "test3 NOT ok"
        return
    x = q.dequeue()
    if x != 1 or q.head != 0:
        print "test3 NOT ok"
        return
    print "test3 OK"
    
test1()
test2()
test3()
