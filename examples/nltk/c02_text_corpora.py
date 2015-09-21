"""
Book: Natural Language Processing with Python
Topic: Accessing Text Corpora

Author: Kelly Chan
Date: Apr 26 2014

"""

import nltk
from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import nps_chat
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.corpus import inaugural
from nltk.corpus import udhr

from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import BracketParseCorpusReader



def gutenberg():

    emma = nltk.corpus.gutenberg.words('austen-emma.txt')
    print len(emma)

    print gutenberg.fileids()
    emma = gutenberg.words('austen-emma.txt')

    macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
    macbeth_sentences[1037]
    longest_len = max([len(s) for s in macbeth_sentences])
    [s for s in macbeth_sentences if len(s) == longest_len]

    for fileid in gutenberg.fileids():
        num_chars = len(gutenberg.raw(fileid))
        num_words = len(gutenberg.words(fileid))
        num_sents = len(gutenberg.sents(fileid))
        num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
        print int(num_chars/num_words), int(num_words/num_sents), int(num_words/num_vocab), fileid


def webtext():

    for fileid in webtext.fileids():
        print fileid, webtext.raw(fileid)[:65], '...'


def chat():
    chatroom = nps_chat.posts('10-19-20s_706posts.xml')
    chatroom[123]


def brown():

    brown.categories()
    brown.words(categories='news')
    brown.words(fileids=['cg22'])
    brown.sents(categories=['news', 'editorial', 'reviews'])

    news_text = brown.words(categories='news')
    fdist = nltk.FreqDist([w.lower() for w in news_text])
    modals = ['can', 'could', 'may', 'might', 'must', 'will']
    for m in modals:
        print m + ':', fdist[m],


    cfd = nltk.ConditionalFreqDist(
            (genre, word)
            for genre in brown.categories()
            for word in brown.words(categories=genre))
    genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
    modals = ['can', 'could', 'may', 'might', 'must', 'will']
    cfd.tabulate(conditions=genres, samples=modals)


def reuters():

    reuters.fileids()
    reuters.categories()

    reuters.categories('training/9865')
    reuters.categories(['training/9865', 'training/9880'])
    reuters.fileids('barley')
    reuters.fileids(['barley', 'corn'])

    reuters.words('training/9865')[:14]
    reuters.words(['training/9865', 'training/9880'])
    reuters.words(categories='barley')
    reuters.words(categories=['barley', 'corn'])


def inaugural():

    inaugural.fileids()
    [fileid[:4] for fileid in inaugural.fileids()]

    cfd = nltk.ConditionalFreqDist(
            (target, fileid[:4])
            for fileid in inaugural.fileids()
            for w in inaugural.words(fileid)
            for target in ['america', 'citizen']
            if w.lower().startswith(target))
    cfd.plot()


def multiLanguages():

    nltk.corpus.cess_esp.words()
    nltk.corpus.floresta.words()
    nltk.corpus.indian.words('hindi.pos')
    nltk.corpus.udhr.fileids()
    nltk.corpus.udhr.words('Javanese-Latin1')[11:]

    languages = ['Chickasaw', 'English', 'German_Deutsch',
            'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
    cfd = nltk.ConditionalFreqDist(
            (lang, len(word))
            for lang in languages
            for word in udhr.words(lang + '-Latin1'))
    cfd.plot(cumulative=True)
    
def structure():

    raw = gutenberg.raw("burgess-busterbrown.txt")
    raw[1:20]

    words = gutenberg.words("burgess-busterbrown.txt")
    words[1:20]

    sents = gutenberg.sents("burgess-busterbrown.txt")
    sents[1:20]


def loadCorpora():

    corpus_root = '/usr/share/dict'
    wordlists = PlaintextCorpusReader(corpus_root, '.*')
    wordlists.fileids()
    wordlists.words('connectives')

    corpus_root = r"C:\corpora\penntreebank\parsed\mrg\wsj"
    file_pattern = r".*/wsj_.*\.mrg" 
    ptb = BracketParseCorpusReader(corpus_root, file_pattern)
    ptb.fileids()
    len(ptb.sents())
    ptb.sents(fileids='20/wsj_2013.mrg')[19]






if __name__ == '__main__':


