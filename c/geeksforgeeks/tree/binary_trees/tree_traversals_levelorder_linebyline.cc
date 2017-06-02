#include <queue>
#include <iostream>

struct node
{
  int data;
  struct node * left;
  struct node * right;
};


void printLevelOrder (node * root)
{
  if (root == NULL)
    return;

  queue<node *> q;
  q.push(root);

  while (1)
  {
    int nodeCount = q.size();
    if (nodeCount == 0)
      break;

    while (nodeCount > 0)
    {
      node * node = q.front();
      std::cout << node->data << " ";
      q.pop();
      
      if (node->left != NULL)
        q.push(node->left);

      if (node->right != NULL)
        q.push(node->right);

      nodeCount--;
    }
    std::cout << std::endl;
  }
}


node * newNode (int data)
{
  node * temp = new node;
  temp->data = data;
  temp->left = NULL;
  temp->right = NULL;

  return temp;
}


int main()
{
  node * root = newNode(1);
  root->left = newNode(2);
  root->right = newNode(3);
  root->left->left = newNode(4);
  root->left->right = newNode(5);
  root->right->right = newNode(6);

  printLevelOrder (root);

  return 0;
}
