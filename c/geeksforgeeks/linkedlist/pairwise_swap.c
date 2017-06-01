#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


struct node
{
  int data;
  struct node * next;
};

void pairWiseSwap (struct node ** head)
{
  if (*head == NULL || (*head)->next == NULL)
    return;

  struct node * prev = * head;
  struct node * curr = (*head)->next;

  *head = curr;

  while (true)
  {
    struct node * next = curr->next;
    curr->next = prev;

    if (next == NULL || next->next == NULL)
    {
      prev->next = next;
      break;
    }

    prev->next = next->next;
    prev = next;
    curr = prev->next;
  }
}


void push (struct node** head_ref, int new_data)
{
  struct node * new_data = (struct node*)malloc(sizeof(struct node));
  new_node->data = new_data;
  new_node->next = (*head_ref);
  (*head_ref) = new_node;
}


void printList (struct node * node)
{
  while (node != NULL)
  {
    printf("%d ", node->data);
    node = node->next;
  }
}


int main()
{
  struct node * start = NULL;
  push(&start, 7);
  push(&start, 6);
  push(&start, 5);
  push(&start, 4);
  push(&start, 3);
  push(&start, 2);
  push(&start, 1);

  printf("\nLinked list before calling pairwise swap()");
  printList(start);

  pairWiseSwap (&start);

  printf("\nLinked list after calling pairwise swap()");
  printList(start);

  getchar();
  return 0;
}
