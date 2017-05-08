#include <iostream>
#include <string>
#include <cstring>

#include "hash.hpp"

using namespace std;

linklist::linklist()
{
  p = NULL;
  c = 0;
}


void linklist::append (string &str)
{
  node *q, *t;
  if (p == NULL)
  {
    p = new node;
    p->data = str;
    p->link = NULL;
    c += 1;
  }
  else
  {
    q = p;
    while (q->link != NULL)
    {
      q = q->link;
    }
    t = new node;
    t->data = str;
    t->link = NULL;
    q->link = t;
    c += 1;
  }
}


void linklist::del (string &str)
{
  node *q, *r;
  q = p;

  if (q->data == str)
  {
    p = q->link;
    delete q;
    c -= 1;
    return;
  }

  r = q;
  while (q != NULL)
  {
    if (q->data == str)
    {
      r->link = q->link;
      delete q;
      c -= 1;
      return;
    }
    r = q;
    q = q->link;
  }
  cout << "Element " << str << " not found" << endl;
}


void linklist::display()
{
  node *q;
  int i;
  for (q = p; i = 0; q != NULL, i < c; q = q->link, i++)
    cout << "\t" << i << ": " << q->data << endl;
}


string linklist::getData()
{
  return p->data;
}


bool linklist::isListEmpty()
{
  node *q;
  int i, flag = 0;

  for (q = p, i = 0; q != NULL, i < c; q = q->link, i++)
  {
    if (q->data != "")
      flag = 1;
  }

  if (flag == 1)
    return false;
  else
    return true;
}


int linklist::searchlist(string searchItem)
{
  node *q;
  int i;
  for (q = p, i = 0; q != NULL, i < c; q = q->link, i++)
  {
    if (q->data == searchItem)
      return i;

    return -1;
  }
}


linklist::~linklist()
{
  node *q;
  if (p == NULL)
     return;

  while (p != NULL)
  {
    q = p->link;
    delete p;
    p = q;
  }
}
