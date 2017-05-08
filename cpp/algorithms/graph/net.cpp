#include <iostream>


struct Vertex
{
  int data;
  struct Vertex* next;
};


struct Edge
{
  struct Vertex* head;
};


class Network
{
  private:
    int num_vertex;
    struct Edge* edge;

  public:

    Network (int num_vertex)
    {
      this->num_vertex = num_vertex;
      edge = new Edge [num_vertex];
    }


    Vertex* addVertex (int data)
    {
      Vertex* new_vertex = new Vertex;
      new_vertex->data = data;
      new_vertex->next = NULL;
      return new_vertex;
    }

 
    void addEdge (int source, int destination)
    {
      Vertex* new_vertex = addVertex(destination);
      new_vertex->next = edge[source].head;
      edge[source].head = new_vertex;
      
      new_vertex = addVertex(source);
      new_vertex->next = edge[destination].head;
      edge[destination].head = new_vertex;
    }


    void printGraph ()
    {
      for (int i = 0; i < num_vertex; ++i)
      {
        Vertex* vertex = edge[i].head;
        std::cout << "vertex " << i << ": ";
        while (vertex)
        {
          std::cout << "->" << vertex->data;
          vertex = vertex->next;
        }
        std::cout << std::endl;
      }
    }
};


int main()
{
  Network net(3);
  net.addEdge(0, 1);
  net.addEdge(1, 2);

  net.printGraph();

  return 0; 
}
