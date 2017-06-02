#include <string.h>
#include <stdio.h>

typedef union
{
  int units;
  float kgs;
} amount;


typedef struct
{
  char selling[15];
  float unitprice;
  int unittype;
  amount howmuch;
} product;


int main()
{
  product dieselmotorbike;
  product apples;
  product * myebaystore[2];

  int nitems = 2;
  int i;

  strcpy(dieselmotorbike.selling, "A Diesel Motor Cycle");
  dieselmotorbike.unitprice = 5488.00;
  dieselmotorbike.unittype = 1;
  dieselmotorbike.howmuch.units = 4;

  strcpy(apples.selling, "Granny duBois");
  apples.unitprice = 0.78;
  apples.unittype = 2;
  apples.howmuch.kgs = 0.5;

  myebaystore[0] = &dieselmotorbike;
  myebaystore[1] = &apples;

  for (i = 0; i < nitems; ++i)
  {
    printf("\n%s\n", myebaystore[i]->selling);
    switch(myebaystore[i]->unittype)
    {
      case 1:
        printf("We have %d units for sale\n", myebaystore[i]->howmuch.units);
        break;
      case 2:
        printf("We have %f kgs for sale\n", myebaystore[i]->howmuch.kgs);
        break;
    }
  }
}
