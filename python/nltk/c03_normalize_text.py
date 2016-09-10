"""
Book: Natural Language Processing with Python
Topic: Normalizing Text

Author: Kelly Chan
Date: Apr 26 2014

"""

import nltk

def token():

    raw = """DENNIS: Listen, strange women lying in ponds distributing swords
    is no basis for a system of government.  Supreme executive power derives from
    a mandate from the masses, not from some farcical aquatic ceremony."""

    tokens = nltk.word_tokenize(raw)

def stem():

    porter = nltk.PorterStemmer()
    lancaster = nltk.LancasterStemmer()

    [porter.stem(t) for t in tokens]
    [lancaster.stem(t) for t in tokens]


class IndexedText(object):

    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                                 for (i, word) in enumerate(text))

    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = width/4                # words of context
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            rcontext = ' '.join(self._text[i:i+wc])
            ldisplay = '%*s'  % (width, lcontext[-width:])
            rdisplay = '%-*s' % (width, rcontext[:width])
            print ldisplay, rdisplay

    def _stem(self, word):
        return self._stemmer.stem(word).lower()


def lemmatization():
    wnl = nltk.WordNetLemmatizer()
    [wnl.lemmatize(t) for t in tokens]

if __name__ == '__main__':

    porter = nltk.PorterStemmer()
    grail = nltk.corpus.webtext.words('grail.txt')
    text = IndexedText(porter, grail)
    text.concordance('lie')



