# -*- encoding: UTF-8 -*-

import sys
import time
from naoqi import ALProxy

if (len(sys.argv) < 2):
    print "Usage: 'python audioplayer_play.py IP [PORT]'"
    sys.exit(1)

IP = sys.argv[1]
PORT = 9559
if (len(sys.argv) > 2):
    PORT = sys.argv[2]
try:
    aup = ALProxy("ALAudioPlayer", IP, PORT)
except Exception,e:
    print "Could not create proxy to ALAudioPlayer"
    print "Error was: ",e
    sys.exit(1)

#Loads a file and launchs the playing 5 seconds later
fileId = aup.loadFile("/usr/share/naoqi/wav/random.wav")
time.sleep(5)
aup.play(fileId)
