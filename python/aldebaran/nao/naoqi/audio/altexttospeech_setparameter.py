import sys
from naoqi import ALProxy

if (len(sys.argv) < 2):
    print "Usage: 'python texttospeech_setparameter.py IP [PORT]'"
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

#Applies a pitch shifting to the voice
tts.setParameter("pitchShift", 1.5)
#Deactivates double voice
tts.setParameter("doubleVoice", 0.0)

tts.say("Pitch shift and double voice changed")