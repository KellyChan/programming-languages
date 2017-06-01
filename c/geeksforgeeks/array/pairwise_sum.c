#include <stdio.h>

void findNumbers(int array[], int n)
{
  int num[n];

  int b_minus_a = array[n-1] - array[1];

  num[1] = (array[0] + b_minus_a) / 2;
  num[0] = array[0] - num[1];
  
  int i;
  for (i = 1; i < (n-2); ++i)
    num[i+1] = array[i] - num[0];

  for (int i = 0; i < n; ++i)
    printf("%d ", num[i]);
}


int main ()
{
  int array[] = {13, 10, 14, 9, 17, 21, 16, 18, 13, 17};
  int n = 5;

  findNumbers(array, n);

  return n;
}
