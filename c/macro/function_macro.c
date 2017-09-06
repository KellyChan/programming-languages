#include <stdio.h>

#define lang_init() c_init()


void c_init(void)
{
  printf("init\n");
}


int main(void)
{
  printf("main\n");
  lang_init();
}
