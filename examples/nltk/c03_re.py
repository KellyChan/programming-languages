"""
Book: Natural Language Processing with Python
Topic: Regular Expressions for Detecting Word Patterns

Author: Kelly Chan
Date: Apr 26 2014

"""

import re
import nltk
from nltk.corpus import gutenberg, nps_chat
from nltk.corpus import brown

def metaCharacters():
    [w for w in wordlist if re.search('ed$', w)]
    [w for w in wordlist if re.search('^..j..t..$', w)]


def range():
    [w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)]

    chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
    [w for w in chat_words if re.search('^m+i+n+e+$', w)]

    [w for w in chat_words if re.search('^[ha]+$', w)]

    wsj = sorted(set(nltk.corpus.treebank.words()))
    [w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)]

    [w for w in wsj if re.search('^[A-Z]+\$$', w)]
    [w for w in wsj if re.search('^[0-9]{4}$', w)]
    [w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)]
    [w for w in wsj if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)]
    [w for w in wsj if re.search('(ed|ing)$', w)]


def extractWordPieces():

    word = 'supercalifragilisticexpialidocious'
    re.findall(r'[aeiou]', word)
    len(re.findall(r'[aeiou]', word))

    wsj = sorted(set(nltk.corpus.treebank.words()))
    fd = nltk.FreqDist(vs for word in wsj
            for vs in re.findall(r'[aeiou]{2,}', word))
    fd.items()

    regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
    def compress(word):
        pieces = re.findall(regexp, word)
        return ''.join(pieces)

    english_udhr = nltk.corpus.udhr.words('English-Latin1')
    print nltk.tokenwrap(compress(w) for w in english_udhr[:75])


    rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
    cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
    cfd = nltk.ConditionalFreqDist(cvs)
    cfd.tabulate()


    cv_word_pairs = [(cv, w) for w in rotokas_words
            for cv in re.findall(r'[ptksvr][aeiou]', w)]
    cv_index = nltk.Index(cv_word_pairs)
    cv_index['su']
    cv_index['po']

def wordStem():

    def stem(word):
        for suffix in ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']:
            if word.endswith(suffix):
                return word[:-len(suffix)]
        return word

    re.findall(r'^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')
    re.findall(r'^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')
    re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')
    re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processes')
    re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processes')
    re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$', 'language')


    def stem(word):
        regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
        stem, suffix = re.findall(regexp, word)[0]
        return stem

    raw = """DENNIS: Listen, strange women lying in ponds distributing swords
    is no basis for a system of government.  Supreme executive power derives from
    a mandate from the masses, not from some farcical aquatic ceremony."""

    tokens = nltk.word_tokenize(raw)
    [stem(t) for t in tokens]


def searchText():

    moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
    moby.findall(r"<a> (<.*>) <man>")
    chat = nltk.Text(nps_chat.words())
    chat.findall(r"<.*> <.*> <bro>") 
    chat.findall(r"<l.*>{3,}") 

    hobbies_learned = nltk.Text(brown.words(categories=['hobbies', 'learned']))
    hobbies_learned.findall(r"<\w*> <and> <other> <\w*s>")








if __name__ == '__main__':

    wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
