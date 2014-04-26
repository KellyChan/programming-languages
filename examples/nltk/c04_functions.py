"""
Book: Natural Language Processing with Python
Topic: Functions: The Foundation of Structured Programming

Author: Kelly Chan
Date: Apr 26 2014

"""

import re

def get_text(file):
    """Read text from a file, normalizing whitespace and stripping HTML markup."""
    text = open(file).read()
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    return text


def repeat(msg, num):
    return ' '.join([msg] * num)

def monty():
    return "Monty Python"

def my_sort1(mylist):      # good: modifies its argument, no return value
    mylist.sort()

def my_sort2(mylist):      # good: doesn't touch its argument, returns value
    return sorted(mylist)

def set_up(word, properties):
    word = 'lolcat'
    properties.append('noun')
    properties = 5

def tag(word):
    if word in ['a', 'the', 'all']:
        return 'det'
    else:
        return 'noun'

def tag(word):
    assert isinstance(word, basestring), "argument to tag() must be a string"
    if word in ['a', 'the', 'all']:
        return 'det'
    else:
        return 'noun'

def freq_words(url, freqdist, n):
    text = nltk.clean_url(url)
    for word in nltk.word_tokenize(text):
        freqdist.inc(word.lower())
    print freqdist.keys()[:n]


def freq_words(url):
    freqdist = nltk.FreqDist()
    text = nltk.clean_url(url)
    for word in nltk.word_tokenize(text):
        freqdist.inc(word.lower())
    return freqdist

def accuracy(reference, test):
    """
    Calculate the fraction of test items that equal the corresponding reference items.

    Given a list of reference values and a corresponding list of test values,
    return the fraction of corresponding values that are equal.
    In particular, return the fraction of indexes
    {0<i<=len(test)} such that C{test[i] == reference[i]}.
    """

def search1(substring, words):
    result = []
    for word in words:
        if substring in word:
            result.append(word)
    return result

def search2(substring, words):
    for word in words:
        if substring in word:
            yield word

def extract_property(prop):
    return [prop(word) for word in sent]

def permutations(seq):
    if len(seq) <= 1:
        yield seq
    else:
        for perm in permutations(seq[1:]):
            for i in range(len(perm)+1):
                yield perm[:i] + seq[0:1] + perm[i:]


def is_content_word(word):
    return word.lower() not in ['a', 'of', 'the', 'and', 'will', ',', '.']

def repeat(msg='<empty>', num=1):
    return msg * num

def generic(*args, **kwargs):
    print args
    print kwargs

def freq_words(file, min=1, num=10):
    text = open(file).read()
    tokens = nltk.word_tokenize(text)
    freqdist = nltk.FreqDist(t for t in tokens if len(t) >= min)
    return freqdist.keys()[:num]


def freq_words(file, min=1, num=10, verbose=False):
    freqdist = FreqDist()
    if verbose: print "Opening", file
    text = open(file).read()
    if verbose: print "Read in %d characters" % len(file)
    for word in nltk.word_tokenize(text):
        if len(word) >= min:
            freqdist.inc(word)
            if verbose and freqdist.N() % 100 == 0: print "."
    if verbose: print
    return freqdist.keys()[:num]


if __name__ == '__main__':

    monty = 'Monty Python'
    repeat(monty, 3)
