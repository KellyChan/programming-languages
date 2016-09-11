# -*- encoding: UTF-8 -*-

import sys
from naoqi import ALProxy

if (len(sys.argv) < 2):
    print "Usage: 'python leds_on.py IP [PORT]'"
    sys.exit(1)

IP = sys.argv[1]
PORT = 9559
if (len(sys.argv) > 2):
    PORT = sys.argv[2]
try:
    proxy = ALProxy("ALLeds", IP, PORT)
except Exception,e:
    print "Could not create proxy to ALLeds"
    print "Error was: ",e
    sys.exit(1)

# Example showing how to switch on a group
name = 'FaceLeds'
proxy.on(name)

sys.exit(0)
