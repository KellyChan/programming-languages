#include <stdio.h>
#include <stdlib.h>


typedef struct
{
  int n;
  float * data;
} tensor;


typedef struct
{
  int n;
  tensor * out;
} ultra;


void print(tensor * a)
{
  printf ("%f, %f\n", a->data[0], a->data[1]);
}


int main(void)
{
  ultra * b = (ultra*)malloc(sizeof(ultra));
  b->n = 1;
  b->out = (tensor*)malloc(sizeof(tensor));
  b->out->n = 2;
  b->out->data = (float*)malloc(b->out->n*sizeof(float));  
  b->out->data[0] = 1;
  b->out->data[1] = 2;

  tensor * a;
  a = b->out;
  print(a);

  free (a->data);
  free (a);
}
