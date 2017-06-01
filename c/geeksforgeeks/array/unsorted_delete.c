#include <stdio.h>

int findElement (int array[], int n, int key);


int deleteElement (int array[], int n, int key)
{
  int pos = findElement(array, n, key);

  if (pos == -1)
  {
    printf("Element not found");
  }

  int i;
  for (i = pos; i < n-1; i++)
    array[i] = array[i+1];

  return n-1;
}

int findElement (int array[], int n, int key)
{
  int i;
  for (i = 0; i < n; i++)
  {
    if (array[i] == key)
      return i;
  }
  return -1;
}


int main ()
{
  int i;
  int array[] = {10, 50, 30, 40, 20};

  int n = sizeof(array)/sizeof(array[0]);
  int key = 30;

  printf("Array before deletion\n");
  for (i = 0; i < n; ++i)
    printf("%d ", array[i]);

  n = deleteElement(array, n, key);

  printf("\n\nArray after deletion\n");
  for (i = 0; i < n; ++i)
    printf("%d ", array[i]);
}
