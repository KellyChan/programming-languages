"""
Book: Natural Language Processing with Python
Topic: Further Examples of Supervised Classification

Author: Kelly Chan
Date: Apr 26 2014

"""

import nltk

def sentenceSeg():

    sents = nltk.corpus.treebank_raw.sents()
    tokens = []
    boundaries = set()
    offset = 0
    for sent in nltk.corpus.treebank_raw.sents():
        tokens.extend(sent)
        offset += len(sent)
        boundaries.add(offset-1)

    def punct_features(tokens, i):
        return {'next-word-capitalized': tokens[i+1][0].isupper(),
                'prevword': tokens[i-1].lower(),
                'punct': tokens[i],
                'prev-word-is-one-char': len(tokens[i-1]) == 1}

    featuresets = [(punct_features(tokens, i), (i in boundaries))
            for i in range(1, len(tokens)-1)
            if tokens[i] in '.?!']

    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    nltk.classify.accuracy(classifier, test_set)

    def segment_sentences(words):
        start = 0
        sents = []
        for i, word in enumerate(words):
            if word in '.?!' and classifier.classify(punct_features(words, i)) == True:
                sents.append(words[start:i+1])
                start = i+1
        if start < len(words):
            sents.append(words[start:])
        return sents


def identifyDialogueActTypes():

    posts = nltk.corpus.nps_chat.xml_posts()[:10000]

    def dialogue_act_features(post):
        features = {}
        for word in nltk.word_tokenize(post):
            features['contains(%s)' % word.lower()] = True
        return features

    featuresets = [(dialogue_act_features(post.text), post.get('class'))
            for post in posts]

    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print nltk.classify.accuracy(classifier, test_set)



def recognizeTextualEntailment():

    def rte_features(rtepair):
        extractor = nltk.RTEFeatureExtractor(rtepair)
        features = {}
        features['word_overlap'] = len(extractor.overlap('word'))
        features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
        features['ne_overlap'] = len(extractor.overlap('ne'))
        features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
        return features   

    rtepair = nltk.corpus.rte.pairs(['rte3_dev.xml'])[33]
    extractor = nltk.RTEFeatureExtractor(rtepair)
    print extractor.text_words 
    print extractor.hyp_words
    print extractor.overlap('word')
    print extractor.overlap('ne')
    print extractor.hyp_extra('word')





