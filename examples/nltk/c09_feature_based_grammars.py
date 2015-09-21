"""
Book: Natural Language Processing with Python
Topic: Building Feature Based Grammars

Author: Kelly Chan
Date: Apr 26 2014

"""

def GrammaticalFeatures():

    kim = {'CAT': 'NP', 'ORTH': 'Kim', 'REF': 'k'}
    chase = {'CAT': 'V', 'ORTH': 'chased', 'REL': 'chase'}

    chase['AGT'] = 'sbj'
    chase['PAT'] = 'obj'

    sent = "Kim chased Lee"
    tokens = sent.split()

    lee = {'CAT': 'NP', 'ORTH': 'Lee', 'REF': 'l'}

    def lex2fs(word):
        for fs in [kim, lee, chase]:
            if fs['ORTH'] == word:
                return fs

    subj, verb, obj = lex2fs(tokens[0]), lex2fs(tokens[1]), lex2fs(tokens[2])
    verb['AGT'] = subj['REF']
    verb['PAT'] = obj['REF']

    for k in ['ORTH', 'REL', 'AGT', 'PAT']:
        print "%-5s => %s" % (k, verb[k])

    surprise = {'CAT': 'V', 'ORTH': 'surprised', 'REL': 'surprise',
            'SRC': 'sbj', 'EXP': 'obj'}

    nltk.data.show_cfg('grammars/book_grammars/feat0.fcfg')


    tokens = 'Kim likes children'.split()
    from nltk import load_parser
    cp = load_parser('grammars/book_grammars/feat0.fcfg', trace=2)
    trees = cp.nbest_parse(tokens)



def processFeatureStructure():

    fs1 = nltk.FeatStruct(TENSE='past', NUM='sg')
    print fs1

    fs1 = nltk.FeatStruct(PER=3, NUM='pl', GND='fem')
    print fs1['GND']

    fs1['CASE'] = 'acc'

    fs2 = nltk.FeatStruct(POS='N', AGR=fs1)
    print fs2

    print fs2['AGR']
    print fs2['AGR']['PER']

    print nltk.FeatStruct("[POS='N', AGR=[PER=3, NUM='pl', GND='fem']]")
    print nltk.FeatStruct(name='Lee', telno='01 27 86 42 96', age=33)


    print nltk.FeatStruct("""[NAME='Lee', ADDRESS=(1)[NUMBER=74, STREET='rue Pascal'],
    SPOUSE=[NAME='Kim', ADDRESS->(1)]]""")


    print nltk.FeatStruct("[A='a', B=(1)[C='c'], D->(1), E->(1)]")







