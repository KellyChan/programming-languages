"""
Check if a directed graph contains a cycle

reference:
http://codereview.stackexchange.com/questions/86021/check-if-a-directed-graph-contains-a-cycle
"""

def cyclic(graph):
    """Return True if the directed graph has a cycle.
    graph must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices.

    >>> cyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    >>> cyclic({1: (2,), 2: (3,), 3: (4:)})
    """

    path = set()

    def visit(vertex):
        path.add(vertex)
        for neighbour in graph.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False

    return any(visit(vertex) for vertex in graph)
