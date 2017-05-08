#include <string>

using namespace std;

class linklist
{
  int c;

  struct node
  {
    public:
      string data;
      node *link;
  } *p;

  public:
    linklist();
    ~linklist();

    void append(string &str);
    void del(string &str);
    void display();
    string getData();
    int searchlist(string);
    bool isListEmpty();
};
