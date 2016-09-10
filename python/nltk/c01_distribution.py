"""
Book: Natural Language Processing with Python
Topic: Conditional Frequency Distributions

Author: Kelly Chan
Date: Apr 26 2014

"""

import nltk
from nltk.corpus import brown
from nltk.corpus import inaugural
from nltk.corpus import udhr

def countWords():

    """counting words by genre"""

    cfd = nltk.ConditionalFreqDist(
            (genre, word)
            for genre in brown.categories()
            for word in brown.words(categories=genre))

    genre_word = [(genre, word)
            for genre in ['news', 'romance']
            for word in brown.words(categories=genre)] 

    print len(genre_word)
    print genre_word[:4]
    print genre_word[-4:]

    cfd = nltk.ConditionalFreqDist(genre_word)
    print cfd.conditions()

    cfd['news']
    cfd['romance']
    list(cfd['romance'])
    cfd['romance']['could']


def tabulate():

    cfd = nltk.ConditionalFreqDist(
            (target, fileid[:4])
            for fileid in inaugural.fileids()
            for w in inaugural.words(fileid)
            for target in ['america', 'citizen']
            if w.lower().startswith(target))

    languages = ['Chickasaw', 'English', 'German_Deutsch',
            'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']

    cfd = nltk.ConditionalFreqDist(
            (lang, len(word))
            for lang in languages
            for word in udhr.words(lang + '-Latin1'))

    cfd.tabulate(conditions=['English', 'German_Deutsch'],
            samples=range(10), cumulative=True)


def genRandomText():

    sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven',
            'and', 'the', 'earth', '.']
    nltk.bigrams(sent)

    def generate_model(cfdist, word, num=15):
        for i in range(num):
            print word,
            word = cfdist[word].max()

    text = nltk.corpus.genesis.words('english-kjv.txt')
    bigrams = nltk.bigrams(text)
    cfd = nltk.ConditionalFreqDist(bigrams)
    print cfd['living']
    generate_model(cfd, 'living')


if __name__ == '__main__':

    countWords()
