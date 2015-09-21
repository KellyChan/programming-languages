"""
Book: Natural Language Processing with Python
Topic: Segmentation

Author: Kelly Chan
Date: Apr 26 2014

"""

import nltk
from random import randint


def sentenceSeg():

    len(nltk.corpus.brown.words()) / len(nltk.corpus.brown.sents())

    sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
    sents = sent_tokenizer.tokenize(text)
    pprint.pprint(sents[171:181])

def wordSeg():

    def segment(text, segs):
        words = []
        last = 0
        for i in range(len(segs)):
            if segs[i] == '1':
                words.append(text[last:i+1])
                last = i+1
        words.append(text[last:])
        return words

    def evaluate(text, segs):
        words = segment(text, segs)
        text_size = len(words)
        lexicon_size = len(' '.join(list(set(words))))
        return text_size + lexicon_size

    text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
    seg1 = "0000000000000001000000000010000000000000000100000000000"
    seg2 = "0100100100100001001001000010100100010010000100010010000"
    seg3 = "0000100100000011001000000110000100010000001100010000001"
    
    segment(text, seg1)
    segment(text, seg2)
    segment(text, seg3)

    evaluate(text, seg3)
    evaluate(text, seg2)
    evaluate(text, seg1)

    
def randomWordSeg():

    def flip(segs, pos):
        return segs[:pos] + str(1-int(segs[pos])) + segs[pos+1:]

    def flip_n(segs, n):
        for i in range(n):
            segs = flip(segs, randint(0,len(segs)-1))
        return segs

    def anneal(text, segs, iterations, cooling_rate):
        temperature = float(len(segs))
        while temperature > 0.5:
            best_segs, best = segs, evaluate(text, segs)
            for i in range(iterations):
                guess = flip_n(segs, int(round(temperature)))
                score = evaluate(text, guess)
                if score < best:
                    best, best_segs = score, guess
            score, segs = best, best_segs
            temperature = temperature / cooling_rate
            print evaluate(text, segs), segment(text, segs)
        print
        return segs

    text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
    seg1 = "0000000000000001000000000010000000000000000100000000000"
    anneal(text, seg1, 5000, 1.2)

