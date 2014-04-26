"""
Book: Natural Language Processing with Python
Topic: Recursion in Linguistic Structure

Author: Kelly Chan
Date: Apr 26 2014

"""


def cascadedChunkers():

    grammar = r"""
      NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
      PP: {<IN><NP>}               # Chunk prepositions followed by NP
      VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
      CLAUSE: {<NP><VP>}           # Chunk NP, VP
      """
    cp = nltk.RegexpParser(grammar)

    sentence = [("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),
        ("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]    

    print cp.parse(sentence)


    sentence = [("John", "NNP"), ("thinks", "VBZ"), ("Mary", "NN"),
            ("saw", "VBD"), ("the", "DT"), ("cat", "NN"), ("sit", "VB"),
            ("on", "IN"), ("the", "DT"), ("mat", "NN")]

    print cp.parse(sentence)

    cp = nltk.RegexpParser(grammar, loop=2)
    print cp.parse(sentence)



def tree():

    tree1 = nltk.Tree('NP', ['Alice'])
    print tree1

    tree2 = nltk.Tree('NP', ['the', 'rabbit'])
    print tree2

    tree3 = nltk.Tree('VP', ['chased', tree2])
    tree4 = nltk.Tree('S', [tree1, tree3])
    print tree4

    print tree4[1]
    tree4[1].node
    tree4.leaves()
    tree4[1][1][1]
    tree3.draw()  


def traverse(t):
    try:
        t.node
    except AttributeError:
        print t,
    else:
        # Now we know that t.node is defined
        print '(', t.node,
        for child in t:
            traverse(child)
        print ')',


def namedEntityRecognition():

    sent = nltk.corpus.treebank.tagged_sents()[22]
    print nltk.ne_chunk(sent, binary=True)
    print nltk.ne_chunk(sent) 


def relationExtraction():

    IN = re.compile(r'.*\bin\b(?!\b.+ing)')
    for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
        for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern = IN):
            print nltk.sem.show_raw_rtuple(rel)




if __name__ == '__main__':

    t = nltk.Tree('(S (NP Alice) (VP chased (NP the rabbit)))')
    traverse(t)


