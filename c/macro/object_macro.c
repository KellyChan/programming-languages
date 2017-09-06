#include <stdio.h>

#define BUFFER_SIZE 1024

#define NUMBERS 1,2,3


int main(void)
{
  int x[] = { NUMBERS };
  printf("%d %d %d\n", x[0], x[1], x[2]);
}
