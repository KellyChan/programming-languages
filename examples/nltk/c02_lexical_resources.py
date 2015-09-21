"""
Book: Natural Language Processing with Python
Topic: Lexical Resources

Author: Kelly Chan
Date: Apr 26 2014

"""

import nltk
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk.corpus import swadesh
from nltk.corpus import toolbox


def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)


def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)

def puzzle():

    puzzle_letters = nltk.FreqDist('egivrvonl')
    obligatory = 'r'
    wordlist = nltk.corpus.words.words()

    [w for w in wordlist if len(w) >= 6 
            and obligatory in w
            and nltk.FreqDist(w) <= puzzle_letters]


def names():

    names = nltk.corpus.names
    names.fileids()
    male_names = names.words('male.txt')
    female_names = names.words('female.txt')

    [w for w in male_names if w in female_names]

    cfd = nltk.ConditionalFreqDist(
            (fileid, name[-1])
            for fileid in names.fileids()
            for name in names.words(fileid))
    cfd.plot()


def pronounce():

    entries = nltk.corpus.cmudict.entries()
    len(entries)

    for entry in entries[39943:39951]:
        print entry

    for word, pron in entries: 
        if len(pron) == 3: 
            ph1, ph2, ph3 = pron
            if ph1 == 'P' and ph3 == 'T':
                print word, ph2,

    syllable = ['N', 'IH0', 'K', 'S']
    [word for word, pron in entries if pron[-4:] == syllable]

    [w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n']
    sorted(set(w[:2] for w, pron in entries if pron[0] == 'N' and w[0] != 'n'))

    def stress(pron):
        return [char for phone in pron for char in phone if char.isdigit()]


    [w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']]
    [w for w, pron in entries if stress(pron) == ['0', '2', '0', '1', '0']]


     p3 = [(pron[0]+'-'+pron[2], word) 
             for (word, pron) in entries
             if pron[0] == 'P' and len(pron) == 3]

     cfd = nltk.ConditionalFreqDist(p3)
     for template in cfd.conditions():
         if len(cfd[template]) > 10:
             words = cfd[template].keys()
             wordlist = ' '.join(words)
             print template, wordlist[:70] + "..."


    prondict = nltk.corpus.cmudict.dict()
    prondict['fire']
    prondict['blog'] = [['B', 'L', 'AA1', 'G']] 
    prondict['blog']

    text = ['natural', 'language', 'processing']
    [ph for w in text for ph in prondict[w][0]]


def compareWordlist():

    swadesh.fileids()
    swadesh.words('en')

    fr2en = swadesh.entries(['fr', 'en'])
    fr2en

    translate = dict(fr2en)
    translate['chien']
    translate['jeter']

    de2en = swadesh.entries(['de', 'en'])    # German-English
    es2en = swadesh.entries(['es', 'en'])    # Spanish-English
    translate.update(dict(de2en))
    translate.update(dict(es2en))
    translate['Hund']
    translate['perro']

    languages = ['en', 'de', 'nl', 'es', 'fr', 'pt', 'la']
    for i in [139, 140, 141, 142]:
        print swadesh.entries(languages)[i]


def toolbox():
    toolbox.entries('rotokas.dic')



if __name__ == '__main__':

    unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))
    unusual_words(nltk.corpus.nps_chat.words())

    stopwords.words('english')
    content_fraction(nltk.corpus.reuters.words())


