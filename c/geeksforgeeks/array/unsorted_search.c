#include <stdio.h>

int findElement (int array[], int n, int key)
{
  int i;
  for (i = 0; i < n; ++i)
  {
    if (array[i] == key)
    {
      return i;
    }
  }
  return -1;
}


int main()
{
  int array[] = {12, 34, 10, 6, 40};
  int n = sizeof(array)/sizeof(array[0]);

  // using a last element as serach element
  int key = 40;
  int position = findElement(array, n, key);

  if (position == -1)
  {
    printf("Element not found");
  }
  else
  {
    printf("Element found at position: %d\n", position+1);
  }

  return 0;
}
