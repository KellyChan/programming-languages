"""
Book: Natural Language Processing with Python
Topic: Supervised Classification

Author: Kelly Chan
Date: Apr 26 2014

"""
import random

import nltk
from nltk.corpus import names
from nltk.classify import apply_features

def gender_features(word):
    return {'last_letter': word[-1]}

def gender_features2(name):
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = name.lower().count(letter)
        features["has(%s)" % letter] = (letter in name.lower())
    return features


def gender_features(word):
    return {'suffix1': word[-1:], \
            'suffix2': word[-2:]}


def documentClassification():

    from nltk.corpus import movie_reviews

    documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)

    all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    word_features = all_words.keys()[:2000]

    def document_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    print document_features(movie_reviews.words('pos/cv957_8737.txt')) 

    featuresets = [(document_features(d), c) for (d,c) in documents]
    train_set, test_set = featuresets[100:], featuresets[:100]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    print nltk.classify.accuracy(classifier, test_set)
    classifier.show_most_informative_features(5)

def partOfSpeechTagging():

    from nltk.corpus import brown

    suffix_fdist = nltk.FreqDist()
    for word in brown.words():
        word = word.lower()
        suffix_fdist.inc(word[-1:])
        suffix_fdist.inc(word[-2:])
        suffix_fdist.inc(word[-3:])

    common_suffixes = suffix_fdist.keys()[:100]
    print common_suffixes 


    def pos_features(word):
        features = {}
        for suffix in common_suffixes:
            features['endswith(%s)' % suffix] = word.lower().endswith(suffix)
        return features

    tagged_words = brown.tagged_words(categories='news')
    featuresets = [(pos_features(n), g) for (n,g) in tagged_words]

    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]

    classifier = nltk.DecisionTreeClassifier.train(train_set)
    nltk.classify.accuracy(classifier, test_set)

    classifier.classify(pos_features('cats'))

    print classifier.pseudocode(depth=4)


def exploitContext():
 	
    def pos_features(sentence, i): [1]
        features = {"suffix(1)": sentence[i][-1:],
                    "suffix(2)": sentence[i][-2:],
                    "suffix(3)": sentence[i][-3:]}
        if i == 0:
            features["prev-word"] = "<START>"
        else:
            features["prev-word"] = sentence[i-1]
        return features

    pos_features(brown.sents()[0], 8)
    tagged_sents = brown.tagged_sents(categories='news')
    featuresets = []
    for tagged_sent in tagged_sents:
        untagged_sent = nltk.tag.untag(tagged_sent)
        for i, (word, tag) in enumerate(tagged_sent):
            featuresets.append( (pos_features(untagged_sent, i), tag) )

    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    nltk.classify.accuracy(classifier, test_set)


def sequenceClassification():

     def pos_features(sentence, i, history): 
         features = {"suffix(1)": sentence[i][-1:],
                     "suffix(2)": sentence[i][-2:],
                     "suffix(3)": sentence[i][-3:]}
         if i == 0:
             features["prev-word"] = "<START>"
             features["prev-tag"] = "<START>"
         else:
             features["prev-word"] = sentence[i-1]
             features["prev-tag"] = history[i-1]
         return features

    class ConsecutivePosTagger(nltk.TaggerI): 

        def __init__(self, train_sents):
            train_set = []
            for tagged_sent in train_sents:
                untagged_sent = nltk.tag.untag(tagged_sent)
                history = []
                for i, (word, tag) in enumerate(tagged_sent):
                    featureset = pos_features(untagged_sent, i, history)
                    train_set.append( (featureset, tag) )
                    history.append(tag)
            self.classifier = nltk.NaiveBayesClassifier.train(train_set)

        def tag(self, sentence):
            history = []
            for i, word in enumerate(sentence):
                featureset = pos_features(sentence, i, history)
                tag = self.classifier.classify(featureset)
                history.append(tag)
            return zip(sentence, history)

    tagged_sents = brown.tagged_sents(categories='news')
    size = int(len(tagged_sents) * 0.1)
    train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]
    tagger = ConsecutivePosTagger(train_sents)
    print tagger.evaluate(test_sents)





if __name__ == '__main__':

    gender_features('Shrek')

    names = ([(name, 'male') for name in names.words('male.txt')] + \
            [(name, 'female') for name in names.words('female.txt')])

    random.shuffle(names)

    featuresets = [(gender_features(n), g) for (n,g) in names]
    train_set, test_set = featuresets[500:], featuresets[:500]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    classifier.classify(gender_features('Neo'))
    classifier.classify(gender_features('Trinity'))

    print nltk.classify.accuracy(classifier, test_set)

    classifier.show_most_informative_features(5)

    train_set = apply_features(gender_features, names[500:])
    test_set = apply_features(gender_features, names[:500])


    featuresets = [(gender_features2(n), g) for (n,g) in names]
    train_set, test_set = featuresets[500:], featuresets[:500]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print nltk.classify.accuracy(classifier, test_set)

    train_names = names[1500:]
    devtest_names = names[500:1500]
    test_names = names[:500]

    train_set = [(gender_features(n), g) for (n,g) in train_names]
    devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
    test_set = [(gender_features(n), g) for (n,g) in test_names]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print nltk.classify.accuracy(classifier, devtest_set)


    errors = []
    for (name, tag) in devtest_names:
        guess = classifier.classify(gender_features(name))
        if guess != tag:
            errors.append( (tag, guess, name) )

    for (tag, guess, name) in sorted(errors): 
        print 'correct=%-8s guess=%-8s name=%-30s' % (tag, guess, name)


    train_set = [(gender_features(n), g) for (n,g) in train_names]
    devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print nltk.classify.accuracy(classifier, devtest_set)





