#include <stdio.h>
#include <stdlib.h>

struct node
{
  int data;
  struct node * left;
  struct node * right;
};


int maxdepth (struct node * node)
{
  if (node == NULL)
    return 0;
  else
  {
    int ldepth = maxdepth(node->left);
    int rdepth = maxdepth(node->right);

    if (ldepth > rdepth)
      return (ldepth + 1);
    else
      return (rdepth + 1);
  }
}


struct node * newNode (int data)
{
  struct node * node = (struct node*)malloc(sizeof(struct node));
  node->data = data;
  node->left = NULL;
  node->right = NULL;

  return (node);
}


int main()
{
  struct node * root = newNode(1);
  root->left = newNode(2);
  root->right = newNode(3);
  root->left->left = newNode(4);
  root->left->right = newNode(5);

  printf("Hight of tree is %d", maxdepth(root));

  getchar();
  return 0;
}
