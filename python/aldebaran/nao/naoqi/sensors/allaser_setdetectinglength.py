# -*- encoding: UTF-8 -*-

import sys
from naoqi import ALProxy

if (len(sys.argv) < 2):
    print "Usage: 'python laser_setdetectinglength.py IP [PORT]'"
    sys.exit(1)

IP = sys.argv[1]
PORT = 9559
if (len(sys.argv) > 2):
    PORT = sys.argv[2]
try:
    laser = ALProxy("ALLaser", IP, PORT)
except Exception,e:
    print "Could not create proxy to ALLaser"
    print "Error was: ",e
    sys.exit(1)

# Set detection threshold at 500/3000 mm
laser.setDetectingLength(500,3000)

print "Set detection threshold at 500/3000 mm"
