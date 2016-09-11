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

#plays a file and get the current position 5 seconds later
fileId = aup.post.playFile("/usr/share/naoqi/wav/random.wav")

time.sleep(5)

#currentPos should be near 5 secs
currentPos = aup.getCurrentPosition(fileId)
