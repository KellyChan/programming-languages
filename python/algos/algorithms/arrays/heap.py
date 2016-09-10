class BinHeap:

    def __init__(self):
        self.heaplist = [0]
        self.heapsize = 0

    def buildheap(self, numlist):
        self.heaplist = [0] + numlist[:]
        self.heapsize = len(numlist)
        self.heapsort(self.heaplist, self.heapsize)

    def heapsort(self, sortinglist, sortsize):
        for index in range(sortsize/2-1, 0, -1):
            sortinglist = self.heapify(sortinglist, index)

    def heapify(self, sortinglist, index):
        left = index * 2
        right = left + 1

        # compare with left, figure out max = [index, left
        if left <= self.heapsize and sortinglist[left] > sortinglist[index]:
            max_index = left
        else:
            max_index = index

        # compare with right, figure out max = [index, left, right]
        if right <= self.heapsize and sortinglist[right] > sortinglist[max_index]:
            max_index = right

        # compare with max
        if index != max_index:
            self._swap(sortinglist, index, max_index)
            self.heapify(sortinglist, max_index)
        
    def _swap(self, sortinglist, index, max_index):
        tmp = sortinglist[index]
        sortinglist[index] = sortinglist[max_index]
        sortinglist[max_index] = tmp

if __name__ == '__main__':

    numlist = [0, 16, 4, 10, 14, 7, 9, 3, 2, 8, 1]

    binheap = BinHeap()
    binheap.buildheap(numlist)
    print binheap.heaplist[:]


