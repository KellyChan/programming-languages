"""
Book: Natural Language Processing with Python
Topic: Evaluation

Author: Kelly Chan
Date: Apr 26 2014

"""

import random
import math
from nltk.corpus import brown


def testSet():

    tagged_sents = list(brown.tagged_sents(categories='news'))
    random.shuffle(tagged_sents)
    size = int(len(tagged_sents) * 0.1)
    train_set, test_set = tagged_sents[size:], tagged_sents[:size]

    file_ids = brown.fileids(categories='news')
    size = int(len(file_ids) * 0.1)
    train_set = brown.tagged_sents(file_ids[size:])
    test_set = brown.tagged_sents(file_ids[:size])

    train_set = brown.tagged_sents(categories='news')
    test_set = brown.tagged_sents(categories='fiction')


def accuracy():

    classifier = nltk.NaiveBayesClassifier.train(train_set) 
    print 'Accuracy: %4.2f' % nltk.classify.accuracy(classifier, test_set) 

def precisionRecall():

    def tag_list(tagged_sents):
        return [tag for sent in tagged_sents for (word, tag) in sent]

    def apply_tagger(tagger, corpus):
        return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]

    gold = tag_list(brown.tagged_sents(categories='editorial'))
    test = tag_list(apply_tagger(t2, brown.tagged_sents(categories='editorial')))
    cm = nltk.ConfusionMatrix(gold, test)
    print cm.pp(sort_by_count=True, show_percents=True, truncate=9) 



def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in nltk.FreqDist(labels)]
    return -sum([p * math.log(p,2) for p in probs])


