"""
Book: Natural Language Processing with Python
Topic: Automatic Tagging

Author: Kelly Chan
Date: Apr 26 2014

"""

import nltk
from nltk.corpus import brown

def defaultTagger():

    brown_tagged_sents = brown.tagged_sents(categories='news')
    brown_sents = brown.sents(categories='news')

    tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
    nltk.FreqDist(tags).max()

    raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
    tokens = nltk.word_tokenize(raw)
    default_tagger = nltk.DefaultTagger('NN')
    default_tagger.tag(tokens)

    default_tagger.evaluate(brown_tagged_sents)


def reTagger():

    patterns = [
         (r'.*ing$', 'VBG'),               # gerunds
         (r'.*ed$', 'VBD'),                # simple past
         (r'.*es$', 'VBZ'),                # 3rd singular present
         (r'.*ould$', 'MD'),               # modals
         (r'.*\'s$', 'NN$'),               # possessive nouns
         (r'.*s$', 'NNS'),                 # plural nouns
         (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
         (r'.*', 'NN')                     # nouns (default)
     ]

    regexp_tagger = nltk.RegexpTagger(patterns)
    regexp_tagger.tag(brown_sents[3])

    regexp_tagger.evaluate(brown_tagged_sents)

def lookupTagger():

    fd = nltk.FreqDist(brown.words(categories='news'))
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    most_freq_words = fd.keys()[:100]
    likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
    baseline_tagger = nltk.UnigramTagger(model=likely_tags)
    baseline_tagger.evaluate(brown_tagged_sents)

    sent = brown.sents(categories='news')[3]
    baseline_tagger.tag(sent)

    baseline_tagger = nltk.UnigramTagger(model=likely_tags,
            backoff=nltk.DefaultTagger('NN'))

    def performance(cfd, wordlist):
        lt = dict((word, cfd[word].max()) for word in wordlist)
        baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
        return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))

    def display():
        import pylab
        words_by_freq = list(nltk.FreqDist(brown.words(categories='news')))
        cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
        sizes = 2 ** pylab.arange(15)
        perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
        pylab.plot(sizes, perfs, '-bo')
        pylab.title('Lookup Tagger Performance with Varying Model Size')
        pylab.xlabel('Model Size')
        pylab.ylabel('Performance')
        pylab.show()   


def NGramTagger():

    from nltk.corpus import brown

    brown_tagged_sents = brown.tagged_sents(categories='news')
    brown_sents = brown.sents(categories='news')
    unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
    unigram_tagger.tag(brown_sents[2007])

    unigram_tagger.evaluate(brown_tagged_sents)

    size = int(len(brown_tagged_sents) * 0.9)
    train_sents = brown_tagged_sents[:size]
    test_sents = brown_tagged_sents[size:]
    unigram_tagger = nltk.UnigramTagger(train_sents)
    unigram_tagger.evaluate(test_sents)

    bigram_tagger = nltk.BigramTagger(train_sents)
    bigram_tagger.tag(brown_sents[2007])

    unseen_sent = brown_sents[4203]
    bigram_tagger.tag(unseen_sent)

    bigram_tagger.evaluate(test_sents)

def combineTagger():
    t0 = nltk.DefaultTagger('NN')
    t1 = nltk.UnigramTagger(train_sents, backoff=t0)
    t2 = nltk.BigramTagger(train_sents, backoff=t1)
    t2.evaluate(test_sents)

def storeTagger():

    from cPickle import dump

    output = open('t2.pkl', 'wb')
    dump(t2, output, -1)
    output.close()

    from cPickle import load
    input = open('t2.pkl', 'rb')
    tagger = load(input)
    input.close()

    text = """The board's action shows what free enterprise
    is up against in our complex maze of regulatory laws ."""
    tokens = text.split()
    tagger.tag(tokens)

def performance():
    cfd = nltk.ConditionalFreqDist(
            ((x[1], y[1], z[0]), z[1])
            for sent in brown_tagged_sents
            for x, y, z in nltk.trigrams(sent))
    ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
    sum(cfd[c].N() for c in ambiguous_contexts) / cfd.N()

    test_tags = [tag for sent in brown.sents(categories='editorial')
            for (word, tag) in t2.tag(sent)]
    gold_tags = [tag for (word, tag) in brown.tagged_words(categories='editorial')]
    print nltk.ConfusionMatrix(gold_tags, test_tags)   








