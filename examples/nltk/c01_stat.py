"""
Book: Natural Language Processing with Python
Topic: Computing with Language: Simple Statistics

Author: Kelly Chan
Date: Apr 26 2014

"""

from __future__ import division

import nltk
from nltk.book import *


def select():

    V = set(text1)
    long_words = [w for w in V if len(w) > 15]
    sorted(long_words)

    fdist5 = FreqDist(text5)
    sorted([w for w in set(text5) if len(w) > 7 and fdist5[w] > 7])

def collocate():

    bigrams(['more', 'is', 'said', 'than', 'done'])
    
    text4.collocations()
    text8.collocations()

def count():

    [len(w) for w in text1]

    fdist = FreqDist([len(w) for w in text1]) 
    fdist.keys()
    fdist.items()
    fdist.max()
    fdist[3]
    fdist.freq(3)

if __name__ == '__main__':

    saying = ['After', 'all', 'is', 'said', 'and', 'done', \
              'more', 'is', 'said', 'than', 'done']

    tokens = set(saying)
    tokens = sorted(tokens)
    print tokens
    print tokens[-2:]

    fdist1 = FreqDist(text1)
    vocabulary1 = fdist1.keys()
    fdist1['whale']
