#include <stdlib.h>

struct list
{
  int x;
  struct list * next;
};


int main (void)
{
  struct list * head = NULL;
  
  int i = 0;
  for (i = 0; i < 6; ++i)
  {
    struct list * node = malloc(sizeof(*node));
    node->next = head;
    head = node;
  } 

  while (head)
  {
    struct list * node = head;
    head = head->next;
    free(node);
  } 

  return 0;
}
