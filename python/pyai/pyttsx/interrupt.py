"""
Interrupting an utterance
==============================================================================

If would stop when the word localtion > 10.

Output:

	word None 0 3
	word None 4 5
	word None 10 5
	word None 16 3
"""

import pyttsx

def onWord(name, location, length):
   print 'word', name, location, length
   if location > 10:
      engine.stop()

engine = pyttsx.init()

engine.connect('started-word', onWord)

engine.say('The quick brown fox jumped over the lazy dog.')
engine.runAndWait()
