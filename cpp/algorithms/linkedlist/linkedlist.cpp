#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <limits>

#define BAD_INPUT_CUTOFF 3

using namespace std;


struct node
{
  int data;
  struct node* next;
};


struct searchDS
{
  int occurence;
  int posit[100];
};


typedef struct node Node;
typedef struct searchDS SD;

Node *start = NULL;
Node *new1;

void print_list(Node*);
int print_no();
int count();
unsigned int getIntInRange(unsigned int min, unsigned int max, const char *prompt);


Node* create_node()
{
  Node* newnode;

  newnode = (Node*)malloc(sizeof(struct node));

  printf("\nEnter the data (numeric only): ");
  scanf("%d", &newnode->data);
  newnode->next = NULL;

  return newnode;
}


int isEmpty()
{
  if (start==NULL)
  {
    printf("\nList is empty");
    return true;
  }
  else
  {
    return false;
  }
}


Node* search (int data)
{
  Node* ptr;
  int flag = 0;
  for (ptr = start; ptr; ptr=ptr->next)
  {
    if (ptr->data == data)
    {
      flag = 1;
      printf("\nData %d found at %x", data, ptr);
    }
  }

  if (flag == 1)
  {
    return ptr;
  }
  else
  {
    printf("\nData %d not found !", data);
    return NULL;
  }
}


