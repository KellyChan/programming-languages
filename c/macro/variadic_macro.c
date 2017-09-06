#include <stdio.h>

#define eprintf(...) fprintf(stderr, __VA_ARGS__)


int main(void)
{
  eprintf("test tests\n");
}
