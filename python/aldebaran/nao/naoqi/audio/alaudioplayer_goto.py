# -*- encoding: UTF-8 -*-

import sys
import time
from naoqi import ALProxy

if (len(sys.argv) < 2):
    print "Usage: 'python audioplayer_goto.py IP [PORT]'"
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

#Launchs the playing of a file, waits for 2 seconds, and goes to the 5th second
fileId = aup.post.playFile("/usr/share/naoqi/wav/random.wav")
time.sleep(1)
aup.goTo(fileId,2)
