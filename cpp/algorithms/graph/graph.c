#include <stdio.h>
#include <stdlib.h>

#include "graph.h"

adjlist_node_p createNode (int v)
{
  adjlist_node_p newNode = (adjlist_node_p)malloc(sizeof(adjlist_node_t));
  if (!newNode)
    err_exit("Unable to allocate memory for new node");

  newNode->vertex = v;
  newNode->next = NULL;

  return newNode;
}


graph_p createGraph (int n, graph_type_e type)
{
  int i;
  graph_p graph = (graph_p)malloc(sizeof(graph_t));
  if (!graph)
  {
    err_exit("Unable to allocate memory for graph");
  }
  graph->num_vertices = n;
  graph->type = type;

  graph->adjListArr = (adjlist_p)malloc(n * sizeof(adjlist_t));
  if (!graph->adjListArr)
    err_exit("Unable to allocate memory for adjacency list array");

  for (i = 0; i < n; i++)
  {
     graph->adjListArr[i].head = NULL;
     graph->adjListArr[i].num_members = 0;
  }

  return graph;
}


void destroyGraph (graph_p graph)
{
  if (graph)
  {
     if (graph->adjListArr)
     {
       int v;
       for (v = 0; v < graph->num_vertices; v++)
       {
         adjlist_node_p adjListPtr = graph->adjListArr[v].head;
         while (adjListPtr)
         {
           adjlist_node_p tmp = adjListPtr;
           adjListPtr = adjListPtr->next;
           free(tmp);
         }
       }
       free(graph->adjListArr);
     }
     free(graph);
  }
}


void addEdge (graph_t *graph, int src, int dest)
{
  adjlist_node_p newNode = createNode(dest);
  newNode->next = graph->adjListArr[src].head;
  graph->adjListArr[src].head = newNode;
  graph->adjListArr[src].num_members++;

  if (graph->type == UNDIRECTED)
  {
    newNode = createNode(src);
    newNode->next = graph->adjListArr[dest].head;
    graph->adjListArr[dest].head = newNode;
    graph->adjListArr[dest].num_members++;
  }
}


void displayGraph (graph_p graph)
{
  int i;
 
  for (i = 0; i < graph->num_vertices; i++)
  {
    adjlist_node_p adjListPtr = graph->adjListArr[i].head;
    printf("\n%d: ", i);

    while (adjListPtr)
    {
      printf("%d->", adjListPtr->vertex);
      adjListPtr = adjListPtr->next;
    }
    printf("NULL\n");
  }
}


int main()
{
  graph_p undir_graph = createGraph (5, UNDIRECTED);
  graph_p dir_graph = createGraph (5, DIRECTED);

  addEdge (undir_graph, 0, 1);
  addEdge (undir_graph, 0, 4);
  addEdge (undir_graph, 1, 2);
  addEdge (undir_graph, 1, 3);
  addEdge (undir_graph, 1, 4);
  addEdge (undir_graph, 2, 3);
  addEdge (undir_graph, 3, 4);

  addEdge (dir_graph, 0, 1);
  addEdge (dir_graph, 0, 4);
  addEdge (dir_graph, 1, 2);
  addEdge (dir_graph, 1, 3);
  addEdge (dir_graph, 1, 4);
  addEdge (dir_graph, 2, 3);
  addEdge (dir_graph, 3, 4);

  printf("\nUNDIRECTED Graph");
  displayGraph(undir_graph);
  destroyGraph(undir_graph);

  printf("\nDIRECTED Graph");
  displayGraph(dir_graph);
  destroyGraph(dir_graph);

  return 0;
}
