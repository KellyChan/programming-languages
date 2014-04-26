"""
Book: Natural Language Processing with Python
Topic: Analyzing Sentence Structure

Author: Kelly Chan
Date: Apr 26 2014

"""

import nltk
from nltk.corpus import treebank

def ambiguity():

    groucho_grammar = nltk.parse_cfg("""
    S -> NP VP
    PP -> P NP
    NP -> Det N | Det N PP | 'I'
    VP -> V NP | VP PP
    Det -> 'an' | 'my'
    N -> 'elephant' | 'pajamas'
    V -> 'shot'
    P -> 'in'
    """)

    sent = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']
    parser = nltk.ChartParser(groucho_grammar)
    trees = parser.nbest_parse(sent)
    for tree in trees:
        print tree

def simpleGrammar():

    grammar1 = nltk.parse_cfg("""
       S -> NP VP
       VP -> V NP | V NP PP
       PP -> P NP
       V -> "saw" | "ate" | "walked"
       NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
       Det -> "a" | "an" | "the" | "my"
       N -> "man" | "dog" | "cat" | "telescope" | "park"
       P -> "in" | "on" | "by" | "with"
       """)

    sent = "Mary saw Bob".split()
    rd_parser = nltk.RecursiveDescentParser(grammar1)
    for tree in rd_parser.nbest_parse(sent):
        print tree


def customGrammar():

    grammar1 = nltk.data.load('file:mygrammar.cfg')
    sent = "Mary saw Bob".split()
    rd_parser = nltk.RecursiveDescentParser(grammar1)
    for tree in rd_parser.nbest_parse(sent):
        print tree


def recursiveSyntacticStructure():

    grammar2 = nltk.parse_cfg("""
    S  -> NP VP
    NP -> Det Nom | PropN
    Nom -> Adj Nom | N
    VP -> V Adj | V NP | V S | V NP PP
    PP -> P NP
    PropN -> 'Buster' | 'Chatterer' | 'Joe'
    Det -> 'the' | 'a'
    N -> 'bear' | 'squirrel' | 'tree' | 'fish' | 'log'
    Adj  -> 'angry' | 'frightened' |  'little' | 'tall'
    V ->  'chased'  | 'saw' | 'said' | 'thought' | 'was' | 'put'
    P -> 'on'
    """)

    rd_parser = nltk.RecursiveDescentParser(grammar1)
    sent = 'Mary saw a dog'.split()
    for t in rd_parser.nbest_parse(sent):
        print t


def shiftParser():

    sr_parse = nltk.ShiftReduceParser(grammar1)
    sent = 'Mary saw a dog'.split()
    print sr_parse.parse(sent)

def leftCorner():

    text = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']
    groucho_grammar.productions(rhs=text[1])


def init_wfst(tokens, grammar):
    numtokens = len(tokens)
    wfst = [[None for i in range(numtokens+1)] for j in range(numtokens+1)]
    for i in range(numtokens):
        productions = grammar.productions(rhs=tokens[i])
        wfst[i][i+1] = productions[0].lhs()
    return wfst

def complete_wfst(wfst, tokens, grammar, trace=False):
    index = dict((p.rhs(), p.lhs()) for p in grammar.productions())
    numtokens = len(tokens)
    for span in range(2, numtokens+1):
        for start in range(numtokens+1-span):
            end = start + span
            for mid in range(start+1, end):
                nt1, nt2 = wfst[start][mid], wfst[mid][end]
                if nt1 and nt2 and (nt1,nt2) in index:
                    wfst[start][end] = index[(nt1,nt2)]
                    if trace:
                        print "[%s] %3s [%s] %3s [%s] ==> [%s] %3s [%s]" % \
                        (start, nt1, mid, nt2, end, start, index[(nt1,nt2)], end)
    return wfst

def display(wfst, tokens):
    print '\nWFST ' + ' '.join([("%-4d" % i) for i in range(1, len(wfst))])
    for i in range(len(wfst)-1):
        print "%d   " % i,
        for j in range(1, len(wfst)):
            print "%-4s" % (wfst[i][j] or '.'),
        print




def dependencyGrammar():

    groucho_dep_grammar = nltk.parse_dependency_grammar("""
    'shot' -> 'I' | 'elephant' | 'in'
    'elephant' -> 'an' | 'in'
    'in' -> 'pajamas'
    'pajamas' -> 'my'
    """)

    print groucho_dep_grammar

    pdp = nltk.ProjectiveDependencyParser(groucho_dep_grammar)
    sent = 'I shot an elephant in my pajamas'.split()
    trees = pdp.parse(sent)
    for tree in trees:
        print tree


def treeBank():

    t = treebank.parsed_sents('wsj_0001.mrg')[0]
    print t


def filter(tree):
    child_nodes = [child.node for child in tree
                   if isinstance(child, nltk.Tree)]
    return  (tree.node == 'VP') and ('S' in child_nodes)


def give(t):
    return t.node == 'VP' and len(t) > 2 and t[1].node == 'NP'\
           and (t[2].node == 'PP-DTV' or t[2].node == 'NP')\
           and ('give' in t[0].leaves() or 'gave' in t[0].leaves())
def sent(t):
    return ' '.join(token for token in t.leaves() if token[0] not in '*-0')

def print_node(t, width):
        output = "%s %s: %s / %s: %s" %\
            (sent(t[0]), t[1].node, sent(t[1]), t[2].node, sent(t[2]))
        if len(output) > width:
            output = output[:width] + "..."
        print output


if __name__ == '__main__':

    [subtree for tree in treebank.parsed_sents()
            for subtree in tree.subtrees(filter)]

    entries = nltk.corpus.ppattach.attachments('training')
    table = nltk.defaultdict(lambda: nltk.defaultdict(set))
    for entry in entries:
        key = entry.noun1 + '-' + entry.prep + '-' + entry.noun2
        table[key][entry.attachment].add(entry.verb)

    for key in sorted(table):
        if len(table[key]) > 1:
            print key, 'N:', sorted(table[key]['N']), 'V:', sorted(table[key]['V'])

    nltk.corpus.sinica_treebank.parsed_sents()[3450].draw()  






