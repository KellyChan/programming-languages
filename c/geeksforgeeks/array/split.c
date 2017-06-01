#include <stdio.h>


int find_split_point(int array[], int n)
{
  int left_sum = 0;

  int i, j;
  for (i = 0; i < n; ++i)
  {
    left_sum += array[i];

    int right_sum = 0;
    for (j = i+1; j < n; j++)
      right_sum += array[j];

    if (left_sum == right_sum)
      return i+1;
  }

  return -1;
}


int find_split_point_2 (int array[], int n)
{
  int left_sum = 0;
  int i;
  for (i = 0; i < n; ++i)
    left_sum += array[i];

  int right_sum = 0;
  for (int i = n-1; i >= 0; i--)
  {
    right_sum += array[i];
    left_sum -= array[i];
    if (right_sum == left_sum)
      return i;
  }

  return -1;
}


void print_two_parts (int array[], int n)
{
  int split_point = find_split_point(array, n);

  if (split_point == -1 || split_point == n)
  {
    printf("Not possible\n");
  }

  int i;
  for (i = 0; i < n; ++i)
  {
    if (split_point == i)
      printf("\n");
    printf("%d ", array[i]);
  }
}


int main()
{
  int array[] = {1, 2, 3, 4, 5, 5};
  int n = sizeof(array)/sizeof(array[0]);

  print_two_parts(array, n);

  return 0;
}
