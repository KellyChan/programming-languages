def max_heapify(li, i, heap_size):
  l = i * 2 #left(i)
  r = l + 1 #right(i)
  if l <= heap_size and li[l] > li[i]:
    largest = l
  else:
    largest = i
  if r <= heap_size and li[r] > li[largest]:
    largest = r
  if i != largest:
    li[i], li[largest] = li[largest], li[i]
    max_heapify(li, largest, heap_size)

def build_max_heap(li, heap_size):
  for i in range(heap_size/2 - 1, 0, -1):
    max_heapify(li, i, heap_size) 

def heap_sort(li, heap_size):
  build_max_heap(li, heap_size)
  for i in range(len(li[1:]), 1, -1):
    li[1], li[i] = li[i], li[1]
    heap_size = heap_size - 1
    max_heapify(li, 1, heap_size)

def main():
  # we shall consider the list from element 1, not 0
  li = [0, 16, 4, 10, 14, 7, 9, 3, 2, 8, 1]
  heap_size = len(li[1:])
  heap_sort(li, heap_size)
  print li[1:]

if __name__ == "__main__":
  main()
