"""
Book: Natural Language Processing with Python
Topic: Information Extraction

Author: Kelly Chan
Date: Apr 26 2014

"""

import nltk
from nltk.corpus import conll2000

def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences] 
    sentences = [nltk.pos_tag(sent) for sent in sentences] 

def chunk():

    sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), 
            ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]

    grammar = "NP: {<DT>?<JJ>*<NN>}"

    cp = nltk.RegexpParser(grammar)
    result = cp.parse(sentence) 
    print result

    result.draw()

def reChunk():

    grammar = r"""
     NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and nouns
          {<NNP>+}                # chunk sequences of proper nouns
    """
    cp = nltk.RegexpParser(grammar)

    sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"),
            ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]

    print cp.parse(sentence)


    nouns = [("money", "NN"), ("market", "NN"), ("fund", "NN")]
    grammar = "NP: {<NN><NN>}  # Chunk two consecutive nouns"
    cp = nltk.RegexpParser(grammar)
    print cp.parse(nouns)


def exploreTextCorpora():

    cp = nltk.RegexpParser('CHUNK: {<V.*> <TO> <V.*>}')
    brown = nltk.corpus.brown
    for sent in brown.tagged_sents():
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.node == 'CHUNK': print subtree

def chink():

    grammar = r"""
      NP:
        {<.*>+}          # Chunk everything
        }<VBD|IN>+{      # Chink sequences of VBD and IN
      """

    sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
           ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]

    cp = nltk.RegexpParser(grammar)

    print cp.parse(sentence)


def evaluate():

    text = '''
    he PRP B-NP
    accepted VBD B-VP
    the DT B-NP
    position NN I-NP
    of IN B-PP
    vice NN B-NP
    chairman NN I-NP
    of IN B-PP
    Carlyle NNP B-NP
    Group NNP I-NP
    , , O
    a DT B-NP
    merchant NN I-NP
    banking NN I-NP
    concern NN I-NP
    . . O
    '''

    nltk.chunk.conllstr2tree(text, chunk_types=['NP']).draw()

    print conll2000.chunked_sents('train.txt')[99]
    print conll2000.chunked_sents('train.txt', chunk_types=['NP'])[99]


def simpleEvaluation():

    cp = nltk.RegexpParser("")
    test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
    print cp.evaluate(test_sents)

    grammar = r"NP: {<[CDJNP].*>+}"
    cp = nltk.RegexpParser(grammar)
    print cp.evaluate(test_sents)


class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents): 
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data) 

    def parse(self, sentence): 
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


class ConsecutiveNPChunkTagger(nltk.TaggerI): 

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) 
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train( 
            train_set, algorithm='megam', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI): 

    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    return {"pos": pos}


def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i-1]
    return {"pos": pos, "prevpos": prevpos}


def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i-1]
    return {"pos": pos, "word": word, "prevpos": prevpos}


def npchunk_features(sentence, i, history):
    
    word, pos = sentence[i]
    
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i-1]
    
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    return {"pos": pos,
            "word": word,
            "prevpos": prevpos,
            "nextpos": nextpos, [1]
            "prevpos+pos": "%s+%s" % (prevpos, pos),  
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt": tags_since_dt(sentence, i)}  


def tags_since_dt(sentence, i):

    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))



if __name__ == '__main__':

    test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
    train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
    unigram_chunker = UnigramChunker(train_sents)
    print unigram_chunker.evaluate(test_sents)

    postags = sorted(set(pos for sent in train_sents
        for (word,pos) in sent.leaves()))

    print unigram_chunker.tagger.tag(postags)

    bigram_chunker = BigramChunker(train_sents)
    print bigram_chunker.evaluate(test_sents)

    chunker = ConsecutiveNPChunker(train_sents)
    print chunker.evaluate(test_sents)

 
