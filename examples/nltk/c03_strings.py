"""
Book: Natural Language Processing with Python
Topic: Strings: Text Processing at the Lowest Level

Author: Kelly Chan
Date: Apr 26 2014

"""

def operate():

    monty = 'Monty Python'
    circus = "Monty Python's Flying Circus"
    circus = 'Monty Python\'s Flying Circus'
    
    couplet = "Shall I compare thee to a Summer's day?"\
            "Thou are more lovely and more temperate:" 
    print couplet

    'very' + 'very' + 'very'
    'very' * 3 

def printString():

    print monty
    grail = 'Holy Grail'
    print monty + grail
    print monty, grail
    print monty, "and the", grail

def access():

    monty[0]
    monty[3]
    monty[5]
    monty[-1]

    sent = 'colorless green ideas sleep furiously'
    for char in sent:
        print char,

    from nltk.corpus import gutenberg
    raw = gutenberg.raw('melville-moby_dick.txt')
    fdist = nltk.FreqDist(ch.lower() for ch in raw if ch.isalpha())
    fdist.keys()

def accessSubstring():

    monty[6:10]
    monty[-12:-7]
    monty[:5]
    monty[6:]

    phrase = 'And now for something completely different'
    if 'thing' in phrase:
        print 'found "thing"'

    monty.find('Python')


def different():
    query = 'Who knows?'
    beatles = ['John', 'Paul', 'George', 'Ringo']
    query[2]
    beatles[2]
    query[:2]
    beatles[:2]

    query + " I don't"
    beatles + 'Brian'
    beatles + ['Brian']

    beatles[0] = "John Lennon"
    del beatles[-1]




