"""
Book: Natural Language Processing with Python
Topic: Reusing Code

Author: Kelly Chan
Date: Apr 26 2014

"""

from __future__ import division

from textproc import plural

def lexical_diversity(text):
    return len(text) / len(set(text))


def lexical_diversity(my_text_data):

    word_count = len(my_text_data)
    vocab_size = len(set(my_text_data))
    diversity_score = word_count / vocab_size

    return diversity_score

def plural(word):

    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'


if __name__ == '__main__':

    plural('wish')
