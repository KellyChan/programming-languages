#include <stdio.h>
#include <errno.h>


int main()
{
  FILE * fp;
  fp = fopen("test.txt", "r");
  printf("Value of errno: %d\n", errno);
  return 0;
}
