#ifndef GRAPH_H_
#define GRAPH_H_

typedef enum 
{
  UNDIRECTED = 0,
  DIRECTED
} graph_type_e;


// Adjacency list node
typedef struct adjlist_node
{
  int vertex;
  struct adjlist_node *next;
} adjlist_node_t, *adjlist_node_p;


// adjacency list
typedef struct adjlist
{
  int num_members;
  adjlist_node_t *head;
} adjlist_t, *adjlist_p;


// graph
typedef struct graph
{
  graph_type_e type;
  int num_vertices;
  adjlist_p adjListArr;
} graph_t, *graph_p;


__inline void err_exit(char* msg)
{
  printf("[Fatal Error]: %s \nExiting ... \n", msg);
  exit(1);
}

#endif

