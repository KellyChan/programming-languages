# -*- encoding: UTF-8 -*-
from naoqi import ALProxy
import sys


def main(robotIP):
    PORT = 9559

    # Create proxy to ALMemory
    try:
        memoryProxy = ALProxy("ALMemory", robotIP, PORT)
    except Exception, e:
        print "Could not create proxy to ALMemory"
        print "Error was: ", e

    # Create proxy to ALSonar
    try:
        sonarProxy = ALProxy("ALSonar", robotIP, PORT)
    except Exception, e:
        print "Could not create proxy to ALSonar"
        print "Error was: ", e

    # Subscribe to sonars, this will launch sonars (at hardware level)
    # and start data acquisition.
    sonarProxy.subscribe("myApplication")

    # Now you can retrieve sonar data from ALMemory.
    # Get sonar left first echo (distance in meters to the first obstacle).
    memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value")

    # Same thing for right.
    memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value")

    # Unsubscribe from sonars, this will stop sonars (at hardware level)
    sonarProxy.unsubscribe("myApplication")

    # Please read Sonar ALMemory keys section
    # if you want to know the other values you can get.


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python sensors_sonar.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
