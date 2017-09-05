#include <stdio.h>
#include <stdlib.h>

typedef struct
{
  void (*forward)(float, float);
} layer;



void relu_forward (float a, float b)
{
  printf ("relu: %f, %f\n", a, b);
}


void fc_forward (float a, float b)
{
  printf ("fc: %f, %f\n", a, b);
}


int main(void)
{
  layer * layers = (layer*)malloc(2*sizeof(layer));
  layers[0].forward = &relu_forward;
  layers[1].forward = &fc_forward;

  int i;
  for (i = 0; i < 2; ++i)
    layers[i].forward(2, 2);

  free (layers);
}
