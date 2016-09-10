"""
Book: Natural Language Processing with Python
Topic: Formatting: From Lists to Strings

Author: Kelly Chan
Date: Apr 26 2014

"""


import nltk
from nltk.corpus import brown

def list2String():

    silly = ['We', 'called', 'him', 'Tortoise', 'because', 'he', 'taught', 'us', '.']
    ' '.join(silly)
    ';'.join(silly)
    ''.join(silly)

def format():
    word = 'cat'
    sentence = """hello
    world"""

    print word
    print sentence

    fdist = nltk.FreqDist(['dog', 'cat', 'dog', 'cat', 'dog', 'snake', 'dog', 'cat'])
    for word in fdist:
        print word, '->', fdist[word], ';',

    for word in fdist:
         print '%s->%d;' % (word, fdist[word]),

    '%s->%d;' % ('cat', 3)
    '%s->' % 'cat'
    '%d' % 3

    'I want a %s right now' % 'coffee'
    "%s wants a %s %s" % ("Lee", "sandwich", "for lunch")


    template = 'Lee wants a %s right now'
    menu = ['sandwich', 'spam fritter', 'pancake']
    for snack in menu:
        print template % snack

def lining():
    
    '%6s' % 'dog'
    '%-6s' % 'dog'
    
    width = 6
    '%-*s' % (width, 'dog')

    count, total = 3205, 9375
    "accuracy for %d words: %2.4f%%" % (total, 100 * count / total)

    '%*s' % (15, "Monty Python")

def tabulate(cfdist, words, categories):

    print '%-16s' % 'Category',
    for word in words:                                  # column headings
        print '%6s' % word,
    print
    for category in categories:
        print '%-16s' % category,                       # row heading
        for word in words:                              # for each word
            print '%6d' % cfdist[category][word],       # print table cell
        print                                           # end the row

    cfd = nltk.ConditionalFreqDist(
            (genre, word)
            for genre in brown.categories()
            for word in brown.words(categories=genre))
    genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
    modals = ['can', 'could', 'may', 'might', 'must', 'will']
    tabulate(cfd, modals, genres)


def toFile():
    output_file = open('output.txt', 'w')
    words = set(nltk.corpus.genesis.words('english-kjv.txt'))
    for word in sorted(words):
        output_file.write(word + "\n")

    len(words)
    str(len(words))
    output_file.write(str(len(words)) + "\n")
    output_file.close()

def wrapping():

    saying = ['After', 'all', 'is', 'said', 'and', 'done', ',',
            'more', 'is', 'said', 'than', 'done', '.']

    for word in saying:
        print word, '(' + str(len(word)) + '),',

    from textwrap import fill
    format = '%s (%d),'
    pieces = [format % (word, len(word)) for word in saying]
    output = ' '.join(pieces)
    wrapped = fill(output)
    print wrapped



