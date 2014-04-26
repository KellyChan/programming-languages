"""
Book: Natural Language Processing with Python
Topic: Texts as Lists of Words

Author: Kelly Chan
Date: Apr 26 2014

"""

from __future__ import division

import nltk
from nltk.book import *


def index():

    text4[173]
    text4.index('awaken')

    text5[16715:16735]
    text6[1600:1625]

    sent = ['word1', 'word2', 'word3', 'word4', 'word5', \
            'word6', 'word7', 'word8', 'word9', 'word10']

    sent[0]
    sent[9]


def lexical_diversity(text):
    return len(text) / len(set(text))

def percentage(count, total):
    return 100 * count / total

if __name__ == '__main__':

    sent1 = ['Call', 'me', 'Ishmael', '.']
    print sent1
    print len(sent1)
    print lexical_diversity(sent1)

    print sent1.append("Some")

    

