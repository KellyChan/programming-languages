#include <stdio.h>
#include <stdlib.h>

typedef struct PString
{
  char * chars;
  int (*length)();
} PString;


int length(PString *self)
{
  return strlen(self->chars);
}


PString *initializeString (int n)
{
  PString *str = malloc(sizeof(PString));

  str->chars = malloc(sizeof(char)*n);
  str->length = length;
  return str;
}


int main()
{
  PString *p = initializeString(30);
  strcpy(p->chars, "Hello");
  printf("%d\n", p->length(p));
  return 0;
}
