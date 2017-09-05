#include <stdio.h>

void foo (int x)
{
  printf("%d\n", x); 
}


int main(void)
{
  void (*dummy)(int);
  dummy = &foo;
  dummy(2);
  dummy(4);

  return 0;
}
