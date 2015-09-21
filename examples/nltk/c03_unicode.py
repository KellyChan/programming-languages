"""
Book: Natural Language Processing with Python
Topic: Text Processing with Unicode

Author: Kelly Chan
Date: Apr 26 2014

"""

import re
import codecs
import unicodedata
import nltk

def loadText():
    path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
    f = codecs.open(path, encoding='latin2')

    for line in f:
        line = line.strip()
        print line.encode('unicode_escape')

    ord('a')
    a = u'\u0061'
    print a

    nacute = u'\u0144'
    print nacute
    nacute_utf = nacute.encode('utf8')
    print repr(nacute_utf)


def loadUnicode():
    lines = codecs.open(path, encoding='latin2').readlines()
    line = lines[2]
    print line.encode('unicode_escape')

    for c in line:
        if ord(c) > 127:
            print '%r U+%04x %s' % (c.encode('utf8'), ord(c), unicodedata.name(c))

    line.find(u'zosta\u0142y')
    line = line.lower()
    print line.encode('unicode_escape')

    m = re.search(u'\u015b\w*', line)
    m.group()

    nltk.word_tokenize(line)  





