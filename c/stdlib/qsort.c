#include <stdio.h>
#include <stdlib.h>

int values [] = {88, 56, 100, 2, 25};

int cmpfunc (const void * a, const void * b)
{
  return (*(int*)a - *(int*)b);
}


int main ()
{
  int n;

  printf("pre sort:\n");
  for (n = 0; n < 5; ++n)
    printf("%d ", values[n]);

  qsort(values, 5, sizeof(int), cmpfunc);

  printf("post sort:\n");
  for (n = 0; n < 5; ++n)
    printf("%d ", values[n]);
}
