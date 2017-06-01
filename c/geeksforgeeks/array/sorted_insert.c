#include <stdio.h>

int insertSorted (int array[], int n, int key, int capacity)
{
  if (n >= capacity)
    return n;

  int i;
  for (i = n - 1; (array[i] > key && i >= 0); i--)
    array[i+1] = array[i];

  array[i+1] = key;

  return (n+1);
}


int main()
{
  int array[20] = {12, 16, 20, 40, 50, 70};
  int capacity = sizeof(array)/sizeof(array[0]);
  int n = 6;
  int i, key = 26;

  printf("\nBefore Insertion: ");
  for (i = 0; i < n; ++i)
    printf("%d ", array[i]);

  n = insertSorted(array, n, key, capacity);

  printf("\nAfter Insertion: ");
  for (i = 0; i < n; ++i)
    printf("%d ", array[i]);

  return 0;
}
