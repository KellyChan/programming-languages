#include <stdio.h>
#include <stdlib.h>


int main()
{
  float * a = (float*)malloc(100*sizeof(float));
  int i;
  for (i = 0; i < 100; ++i)
    a[i] = i;


  float * b = a + 5;
  for (i = 0; i < 5; ++i)
  {
    printf("%f,", b[i]);
  }

  free(a);

  return 0;
}
