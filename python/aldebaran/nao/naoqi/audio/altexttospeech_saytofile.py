import sys
from naoqi import ALProxy

if (len(sys.argv) < 2):
    print "Usage: 'python texttospeech_saytofile.py IP [PORT]'"
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

#Says a test std::string, and save it into a file
tts.sayToFile("This is a sample text, written in a file!", "/tmp/sample_text.raw")

#Says a test std::string, and save it into a file
tts.sayToFile("This is another sample text", "/tmp/sample_text.wav")