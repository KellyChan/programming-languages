import sys
from naoqi import ALProxy

if (len(sys.argv) < 2):
    print "Usage: 'python texttospeech_getavailablelanguages.py IP [PORT]'"
    sys.exit(1)

IP = sys.argv[1]
PORT = 9559
if (len(sys.argv) > 2):
    PORT = sys.argv[2]
try:
    tts = ALProxy("ALTextToSpeech", IP, PORT)
except Exception,e:
    print "Could not create proxy to ALTextToSpeech"
    print "Error was: ",e
    sys.exit(1)

lang = tts.getAvailableLanguages();
print "Available languages: " + str(lang)