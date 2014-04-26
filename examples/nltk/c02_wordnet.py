"""
Book: Natural Language Processing with Python
Topic: WordNet

Author: Kelly Chan
Date: Apr 26 2014

"""

import nltk
from nltk.corpus import wordnet as wn

def wordnet():

    wn.synsets('motorcar')
    wn.synset('car.n.01').lemma_names
    wn.synset('car.n.01').definition
    wn.synset('car.n.01').examples

    wn.synset('car.n.01').lemmas
    wn.lemma('car.n.01.automobile') 
    wn.lemma('car.n.01.automobile').synset
    wn.lemma('car.n.01.automobile').name

    wn.synsets('car')
    for synset in wn.synsets('car'):
        print synset.lemma_names

    wn.lemmas('car')


def hirarchy():

    motorcar = wn.synset('car.n.01')
    types_of_motorcar = motorcar.hyponyms()
    types_of_motorcar[26]

    sorted([lemma.name for synset in types_of_motorcar for lemma in synset.lemmas])

    motorcar.hypernyms()
    paths = motorcar.hypernym_paths()
    len(paths)

    [synset.name for synset in paths[0]]
    [synset.name for synset in paths[1]]
    motorcar.root_hypernyms()


def relations():

    wn.synset('tree.n.01').part_meronyms()
    wn.synset('tree.n.01').substance_meronyms()
    wn.synset('tree.n.01').member_holonyms()

    for synset in wn.synsets('mint', wn.NOUN):
        print synset.name + ':', synset.definition

    wn.synset('mint.n.04').part_holonyms()
    wn.synset('mint.n.04').substance_holonyms()

    wn.synset('walk.v.01').entailments()
    wn.synset('eat.v.01').entailments()
    wn.synset('tease.v.03').entailments()

    wn.lemma('supply.n.02.supply').antonyms()
    wn.lemma('rush.v.01.rush').antonyms()
    wn.lemma('horizontal.a.01.horizontal').antonyms()
    wn.lemma('staccato.r.01.staccato').antonyms()

def semanticSimilarity():

    right = wn.synset('right_whale.n.01')
    orca = wn.synset('orca.n.01')
    minke = wn.synset('minke_whale.n.01')
    tortoise = wn.synset('tortoise.n.01')
    novel = wn.synset('novel.n.01')
    right.lowest_common_hypernyms(minke)
    right.lowest_common_hypernyms(orca)
    right.lowest_common_hypernyms(tortoise)
    right.lowest_common_hypernyms(novel)

    wn.synset('baleen_whale.n.01').min_depth()
    wn.synset('whale.n.02').min_depth()
    wn.synset('vertebrate.n.01').min_depth()
    wn.synset('entity.n.01').min_depth()


    right.path_similarity(minke)
    right.path_similarity(orca)
    right.path_similarity(tortoise)
    right.path_similarity(novel)



