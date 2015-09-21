"""
Book: Natural Language Processing with Python
Topic: A Sample of Python Libraries

Author: Kelly Chan
Date: Apr 26 2014

"""

def matplotlib():

    colors = 'rgbcmyk' # red, green, blue, cyan, magenta, yellow, black
    def bar_chart(categories, words, counts):
        "Plot a bar chart showing counts for each word by category"
        import pylab
        ind = pylab.arange(len(words))
        width = 1 / (len(categories) + 1)
        bar_groups = []
        for c in range(len(categories)):
            bars = pylab.bar(ind+c*width, counts[categories[c]], width,
                             color=colors[c % len(colors)])
            bar_groups.append(bars)
        pylab.xticks(ind+width, words)
        pylab.legend([b[0] for b in bar_groups], categories, loc='upper left')
        pylab.ylabel('Frequency')
        pylab.title('Frequency of Six Modal Verbs by Genre')
        pylab.show()

def networkX():

    import networkx as nx
    import matplotlib
    from nltk.corpus import wordnet as wn

    def traverse(graph, start, node):
        graph.depth[node.name] = node.shortest_path_distance(start)
        for child in node.hyponyms():
            graph.add_edge(node.name, child.name) [1]
            traverse(graph, start, child) [2]

    def hyponym_graph(start):
        G = nx.Graph() [3]
        G.depth = {}
        traverse(G, start, start)
        return G

    def graph_draw(graph):
        nx.draw_graphviz(graph,
             node_size = [16 * graph.degree(n) for n in graph],
             node_color = [graph.depth[n] for n in graph],
             with_labels = False)
        matplotlib.pyplot.show()

def csv():

    import csv
    input_file = open("lexicon.csv", "rb") 
    for row in csv.reader(input_file): 
        print row

def numpy():

    from numpy import array

    cube = array([ [[0,0,0], [1,1,1], [2,2,2]],
                   [[3,3,3], [4,4,4], [5,5,5]],
                   [[6,6,6], [7,7,7], [8,8,8]] ])

    cube[1,1,1]
    cube[2].transpose()
    cube[2,1:]


    from numpy import linalg
    a=array([[4,0], [3,-5]])
    u,s,vt = linalg.svd(a)
