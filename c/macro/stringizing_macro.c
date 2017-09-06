#include <stdio.h>
#include <stdlib.h>

#define WARN_IF(X) \
  do { if (X == 0)      \
          fprintf(stderr, "Warning: EXP\n"); \
     } \
  while (0)


int main(void)
{
  WARN_IF(0);
}
