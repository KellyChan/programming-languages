"""
Running a driver event loop
==============================================================================

output:

	starting fox
	word fox 0 3
	word fox 4 5
	word fox 10 5
	word fox 16 3
	word fox 20 6
	word fox 27 4
	word fox 32 3
	word fox 36 4
	word fox 41 3
	word fox 44 1
	finishing fox True
	starting dog
	word dog 0 4
	word dog 5 1
	word dog 7 4
	word dog 12 3
	word dog 15 1
	finishing dog True
"""

import pyttsx

def onStart(name):
   print 'starting', name


def onWord(name, location, length):
   print 'word', name, location, length


def onEnd(name, completed):
   print 'finishing', name, completed
   if name == 'fox':
      engine.say('What a lazy dog!', 'dog')
   elif name == 'dog':
      engine.endLoop()


engine = pyttsx.init()

engine.connect('started-utterance', onStart)
engine.connect('started-word', onWord)
engine.connect('finished-utterance', onEnd)

engine.say('The quick brown fox jumped over the lazy dog.', 'fox')
engine.startLoop()
