# -*- encoding: UTF-8 -*-

import sys
from naoqi import ALProxy

if (len(sys.argv) < 2):
    print "Usage: 'python leds_fade.py IP [PORT]'"
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

# Example showing how to fade the ears group to mid-intensity
name = 'EarLeds'
intensity = 0.5
duration = 1.0
proxy.fade(name, intensity, duration)
