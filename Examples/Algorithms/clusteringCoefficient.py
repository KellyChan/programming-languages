def make_link(G, node1, node2):
    if node1 not in G:
        G[node1] = {}
    (G[node1])[node2] = 1
    if node2 not in G:
        G[node2] = {}
    (G[node2])[node1] = 1
    return G
 
flights = [("ORD", "SEA"), ("ORD", "LAX"), ('ORD', 'DFW'), ('ORD', 'PIT'),
           ('SEA', 'LAX'), ('LAX', 'DFW'), ('ATL', 'PIT'), ('ATL', 'RDU'),
           ('RDU', 'PHL'), ('PIT', 'PHL'), ('PHL', 'PVD')]
 
G = {}
for (x,y) in flights: make_link(G,x,y)
 
def clustering_coefficient(G,v):
    neighbors = G[v].keys()
    if len(neighbors) == 1: return -1.0
    links = 0
    for w in neighbors:
        for u in neighbors:
            if u in G[w]: links += 0.5
    return 2.0*links/(len(neighbors)*(len(neighbors)-1))
 
print clustering_coefficient(G,"ORD")
 
total = 0
for v in G.keys():
    total += clustering_coefficient(G,v)
 
print total/len(G)