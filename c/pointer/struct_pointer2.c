#include <stdio.h>
#include <stdlib.h>


typedef struct
{
  char * name;
  void (*init)(float, float);
  void (*forward)(float, float);
  void (*del)();
} layer;


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


int main(void)
{
  layer * layers = (layer*)malloc(2*sizeof(layer));

  layers[0].name = "relu";
  layers[0].init = &relu_init;
  layers[0].forward = &relu_forward;
  
  layers[1].name = "fc";
  layers[1].init = &fc_init;
  layers[1].forward = &fc_forward;

  int i;
  for (i = 0; i < 2; ++i)
  {
    layers[i].init(2, 2);
    layers[i].forward(4, 4);
  }  

  free (layers);
}
