# -*- encoding: UTF-8 -*-

import sys
import time
from naoqi import ALProxy

if (len(sys.argv) < 2):
    print "Usage: 'python audioplayer_playfile.py IP [PORT]'"
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
    
#Launchs the playing of a file
aup.playFile("/usr/share/naoqi/wav/random.wav")

time.sleep(1.0)

#Launchs the playing of a file on the left speaker to a volume of 50%
aup.playFile("/usr/share/naoqi/wav/random.wav",0.5,-1.0)
