import sys
from naoqi import ALProxy

if (len(sys.argv) < 2):
    print "Usage: 'python texttospeech_setvoice.py IP [PORT]'"
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

#Changes the basic voice of the synthesis
tts.setVoice("Kenny22Enhanced")

tts.say("Voice changed to Kenny")