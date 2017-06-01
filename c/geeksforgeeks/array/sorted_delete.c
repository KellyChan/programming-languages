#include <stdio.h>

int binarySearch (int array[], int low, int high, int key)
{
  if (high < low)
    return -1;

  int mid = (low+high)/2;

  if (key == array[mid])
    return mid;

  if (key > array[mid])
    return binarySearch(array, (mid+1), high, key);

  return binarySearch(array, low, (mid-1), key);
}


int deleteElement (int array[], int n, int key)
{
  int pos = binarySearch(array, 0, n-1, key);

  if (pos == -1)
  {
    printf("Element not found");
    return n;
  }

  int i;
  for (i = pos; i < n; ++i)
    array[i] = array[i+1];

  return n-1;
}


int main ()
{
  int i;
  int array[] = {10, 20, 30, 40, 50};

  int n = sizeof(array)/sizeof(array[0]);
  int key = 30;

  printf("Array before deletion\n");
  for (i = 0; i < n; ++i)
    printf("%d ", array[i]);

  n = deleteElement(array, n, key);

  printf("\nArray after deletion:\n");
  for (i = 0; i < n; ++i)
    printf("%d ", array[i]);
}
