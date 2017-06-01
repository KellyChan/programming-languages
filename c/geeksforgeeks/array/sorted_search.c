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


int main()
{
  int array[] = {5, 6, 7, 8, 9, 10};
  int n, key;

  n = sizeof(array)/sizeof(array[0]);
  key = 10;

  printf("Index: %d\n", binarySearch(array, 0, n, key));

  return 0;
}
