"""
Listening for events
==============================================================================

Run

    $ python event.py

Output

	starting None
	word None 0 3
	word None 4 5
	word None 10 5
	word None 16 3
	word None 20 6
	word None 27 4
	word None 32 3
	word None 36 4
	word None 41 3
	word None 44 1
	finishing None True
"""

import pyttsx


def onStart(name):
    print 'starting', name


def onWord(name, location, length):
    """
    Inputs:
        - name:
        - location: word location (column)
        - length: word length

    NOTE: DON'T CHANGE THE VARIABLES.
    """
    print 'word', name, location, length


def onEnd(name, completed):
    print 'finishing', name, completed


engine = pyttsx.init()

engine.connect('started-utterance', onStart)
engine.connect('started-word', onWord)
engine.connect('finished-utterance', onEnd)

engine.say('The quick brown fox jumped over the lazy dog.')
engine.runAndWait()
