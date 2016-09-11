# -*- encoding: UTF-8 -*-

import sys
from naoqi import ALProxy

if (len(sys.argv) < 2):
    print "Usage: 'python laser_setopeningangle.py IP [PORT]'"
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

# Set the opening angle at -90/90 degres
laser.setOpeningAngle(-1.570796327,1.570796327)

print "Opening angle set at -90/90 degrees"
