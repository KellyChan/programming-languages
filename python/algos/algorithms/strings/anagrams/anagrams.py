def possible_anagrams(word):
    """Given a word, return a list of possible anagrams for that word.
    """

    if len(word) == 1:
        yield word

    for i, letter in enumerate(word):
        for s in possible_anagrams(word[i+1:] + word[:i):
            yield '%s%s' % (letter, s)

def anagrams(word):
    """Given a word, return a list of anagrams in an english dictionary.
    """

    words = [w.rstrip() for w in open('WORD.LIST')]
    return [a for a in possible_anagrams(word) if a in words]


#-------------------------------------------------------#

import itertools

def anagrams(word):

    words = [w.rstrip() for w in open('WORD.LST')]
    # reduce duplcations
    words = set([w for w in words if len(w) == len(word)])

    # find all possible anagrams
    comb = set([''.join(w) for w in itertools.permutations(word, len(word))]) 

    return comb.intersection(words)
 

if __name__ == '__main__':

    print anagrams('python')
