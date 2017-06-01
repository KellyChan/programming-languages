#include <stdio.h>


int insertSorted (int array[], int n, int key, int capacity)
{
  if (n >= capacity)
    return n;

  array[n] = key;
  return (n+1);
}


int main ()
{
  int array[20] = {12, 16, 20, 40, 50, 70};
  int capacity = sizeof(array)/sizeof(array[0]);
  int n = 9;
  int key = 26;

  printf("\nBefore Insertion: ");
  int i;
  for (i = 0; i < n; ++i)
    printf("%d ", array[i]);

  n = insertSorted(array, n, key, capacity);

  printf("\nAfter Insertion: ");
  for (i = 0; i < n; ++i)
    printf("%d ", array[i]);

  return 0;
}
