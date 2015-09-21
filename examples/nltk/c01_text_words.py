"""
Book: Natural Language Processing with Python
Topic: Computing with Language: Texts and Words

Author: Kelly Chan
Date: Apr 26 2014

"""

from __future__ import division

import nltk
from nltk.book import *



def search():
    text1.concordance("monstrous")
    text1.similar("monstrous")
    text2.similar("monstrous")
    text2.common_contexts(["monstrous", "very"])

def generate():
    text3.generate()
    

def plot():
    text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])


def count():
    len(text3)
    sorted(set(text3))
    len(set(text3))

    len(text3) / len(set(text3))

    text3.count("smote")
    100 * text4.count('a') / len(text4)


def lexical_diversity(text):
    return len(text) / len(set(text))

def percentage(count, total):
    return 100 * count / total

if __name__ == '__main__':


    print text1
    print text2
    
    lexical_diversity(text3)
    lexical_diversity(text5)
    percentage(text4.count('a'), len(text4))
