"""
Book: Natural Language Processing with Python
Topic: Accessing Text from the Web and from Disk

Author: Kelly Chan
Date: Apr 26 2014

"""

from urllib import urlopen

def loadBook():

    url = "http://www.gutenberg.org/files/2554/2554.txt"
    raw = urlopen(url).read()
    type(raw)
    len(raw)
    raw[:75]

    proxies = {'http': 'http://www.someproxy.com:3128'}
    raw = urlopen(url, proxies=proxies).read()

def loadHTML():
    url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
    html = urlopen(url).read()
    html[:60]

    raw = nltk.clean_html(html)
    tokens = nltk.word_tokenize(raw)
    tokens

    tokens = tokens[96:399]
    text = nltk.Text(tokens)
    text.concordance('gene')

def loadRSS():

    import feedparser
    llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
    llog['feed']['title']

    len(llog.entries)
    post = llog.entries[2]
    post.title

    content = post.content[0].value
    content[:70]

    nltk.word_tokenize(nltk.clean_html(content))
    nltk.word_tokenize(nltk.clean_html(llog.entries[2].content[0].value))

def loadFile():

    f = open('document.txt')
    raw = f.read()

    import os
    os.listdir('.')
    f.read()

    f = open('document.txt', 'rU')
    for line in f:
        print line.strip()


    path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')
    raw = open(path, 'rU').read()


def loadInput():

    s = raw_input("Enter some text: ")
    print "You typed", len(nltk.word_tokenize(s)), "words."




def token():

    tokens = nltk.word_tokenize(raw)
    type(tokens)
    len(tokens)
    tokens[:10]

def text():
    text = nltk.Text(tokens)
    type(text)
    text[1020:1060]
    text.collocations()

def find():
    raw.find("PART I")
    raw.rfind("End of Project Gutenberg's Crime")
    raw = raw[5303:1157681]
    raw.find("PART I")


def pipeline():

    raw = open('document.txt').read()
    type(raw)

    tokens = nltk.word_tokenize(raw)
    type(tokens)

    words = [w.lower() for w in tokens]
    type(words)

    vocab = sorted(set(words))
    type(vocab)

    vocab.append('blog')
    raw.append('blog')

    query = 'Who knows?'
    beatles = ['john', 'paul', 'george', 'ringo']
    query + beatles
