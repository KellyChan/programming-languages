"""
Book: Natural Language Processing with Python
Topic: Analyzing the Meaning of Sentences

Author: Kelly Chan
Date: Apr 26 2014

"""

def firstOrderLogic():

    tlp = nltk.LogicParser(type_check=True)
    parsed = tlp.parse('walk(angus)')
    parsed.argument


    parsed.argument.type
    parsed.function
    parsed.function.type

    sig = {'walk': '<e, t>'}
    parsed = tlp.parse('walk(angus)', sig)
    parsed.function.type

    lp = nltk.LogicParser()
    lp.parse('dog(cyril)').free()

    lp.parse('dog(x)').free()
    lp.parse('own(angus, cyril)').free()
    lp.parse('exists x.dog(x)').free()
    lp.parse('((some x. walk(x)) -> sing(x))').free()
    lp.parse('exists x.own(y, x)').free()
