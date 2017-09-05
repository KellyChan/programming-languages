#include <stdio.h>
#include <stdlib.h>



void relu_init (float a, float b)
{
  printf ("relu_init: %f, %f\n", a, b);
}

void relu_forward (float a, float b)
{
  printf ("relu_forward: %f, %f\n", a, b);
}


void fc_init (float a, float b)
{
  printf ("fc_init: %f, %f\n", a, b);
}

void fc_forward (float a, float b)
{
  printf ("fc_forward: %f, %f\n", a, b);
}


void test_forward (float a, float b, void (*f)(float, float))
{
  f(a, b);
}


int main(void)
{
  test_forward (1, 2, relu_init);
  test_forward (2, 2, relu_forward);
  test_forward (2, 2, fc_forward);
}
