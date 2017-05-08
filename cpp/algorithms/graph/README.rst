ref: http://simplestcodings.blogspot.com/2013/09/graphs.html

categorized by edges:

- directed, weighted edges
- directed, unweighted edges
- undirected, weighted edges
- undirected, unweighted edges

Use of graphs:

- metro rail map
- a maze
- a tournament fixture

Kind of graphs:

- undirected graph
- directed graph
- vertext labeled graph
- edge labeled graph
- cyclic graph
- weighted graph
- directed acyclic graph (graph has no cycles)
- disconnected graph
- mixed graph
- multigraph
- quiver


Representation of graph (Adjacency matrix):

1. for N vertices, an adjacency matrix is an NxN array A such that

- A[i][j] = 1 if there is an edge E(i, j)
- A[i][j] = 0 otherwise

2. for an undirected graph

- A[i][j]  = A[j][i]

3. for weighted graphs

- A[i][j] = weight of the edge, if there is an edge E(i, j)
- A[i][j] = a constant representing no edge (e.g. a very large or very small value)

Representation of graph (Adjacency list):

- every vertext has a linked list of vertices it is connected with

==============================================================================
(Vertex) Graph Search
==============================================================================

ref: https://www.topcoder.com/community/data-science/data-science-tutorials/introduction-to-graphs-and-their-data-structures-section-2/

- graph - linkedlist
- depth first - stack
- breath first - queue

stack  queue    heap     priorityQueue
add()  add()    add()
pop()  push()   pop()
top()  front()  top()
del()  del()    empty()

==============================================================================
(edge) Path Search 
==============================================================================

- shortest path: Dijkstra (Heap), Floyd Warshall
