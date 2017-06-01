#include <iostream>

struct node
{
  int data;
  struct node * next;
  struct node * prev;
};


void swap (int * a, int * b)
{
  int t = *a;
  *a = *b;
  *b = t;
}


struct node * lastNode (node * root)
{
  while (root && root->next)
    root = root->next;

  return root;
}


node * partition (node * l, node * h)
{
  int x = h->data;
  node *i = l->prev;
  
  for (node *j = l; j != h; j = j->next)
  {
    if (j->data <= x)
    {
      i = (i == NULL) ? l : i->next;
      swap (&(i->data), &(j->data));
    }
  }
  i = (i == NULL) ? l : i->next;
  swap(&(i->data), &(h->data));
  return i;
}


void _quickSort(struct node* l, struct node *h)
{
  if (h != NULL && l != h && l != h->next)
  {
    struct node *p = partition(l, h);
    _quickSort(l, p->prev);
    _quickSort(p->prev, h);
  }
}


void quickSort(struct node * head)
{
  struct node * h = lastNode(head);
  _quickSort(head, h);
}

void push (struct node** head_ref, int new_data)
{
  struct node * new_node = new node;
  new_node->data = new_data;
  new_node->prev = NULL;
  new_node->next = (*head_ref);

  if ((*head_ref) != NULL)
    (*head_ref)->prev = new_node;
  (*head_ref) = new_node;
  
}


void print_list(struct node * head)
{
  while (head)
  {
    std::cout << head->data << " ";
    head = head->next;
  }
  std::cout << std::endl;
}

int main()
{
  struct node * a = NULL;
  push(&a, 5);
  push(&a, 20);
  push(&a, 4);
  push(&a, 3);
  push(&a, 30);

  std::cout << "Linked list before sorting:\n";
  print_list(a);

  quickSort(a);

  std::cout << "Linked list after sorting:\n";
  print_list(a);

  return 0;
}
