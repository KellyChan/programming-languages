"""
Book: Natural Language Processing with Python
Topic: Making Decisions and Taking Control

Author: Kelly Chan
Date: Apr 26 2014

"""

from __future__ import division

import nltk
from nltk.book import *


def operate():

    [len(w) for w in text1]
    [w.upper() for w in text1]

    len(text1)
    len(set(text1))
    len(set([word.lower() for word in text1]))

    len(set([word.lower() for word in text1 if word.isalpha()]))

def nest():

    for word in ['Call', 'me', 'Ishmael', '.']:
        print word

def block():

    word = 'cat'
    if len(word) < 5:
        print 'word length is less than 5'
    else:
        print 'word length is greater than or equal to 5'


def loop():

    sent1 = ['Call', 'me', 'Ishmael', '.']
    for xyzzy in sent1:
        if xyzzy.endswith('l'):
            print xyzzy


    for token in sent1:
        if token.islower():
            print token, 'is a lowercase word'
        elif token.istitle():
            print token, 'is a titlecase word'
        else:
            print token, 'is punctuation'




if __name__ == '__name__':

    sorted([w for w in set(text1) if w.endswith('ableness')])
    sorted([term for term in set(text4) if 'gnt' in term])
    sorted([item for item in set(text6) if item.istitle()])
    sorted([item for item in set(sent7) if item.isdigit()])
