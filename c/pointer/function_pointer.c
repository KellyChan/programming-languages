#include <stdio.h>

void foo (int x, float s)
{
  printf("%d %f\n", x, s); 
}


int main(void)
{
  void (*dummy)(int, float);
  dummy = &foo;
  dummy(2, 2);
  dummy(4, 2);

  return 0;
}
