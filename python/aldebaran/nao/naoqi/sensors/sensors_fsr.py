# -*- encoding: UTF-8 -*-
from naoqi import ALProxy
import time
import sys

def main(robotIP):
    PORT = 9559

    # Create proxy to ALMemory
    try:
        memoryProxy = ALProxy("ALMemory", robotIP, PORT)
    except Exception, e:
        print "Could not create proxy to ALMemory"
        print "Error was: ", e

    footContact = True

    footContact = memoryProxy.getData("footContact")

    while footContact:
        footContact = memoryProxy.getData("footContact")
        leftFoot    = memoryProxy.getData("leftFootTotalWeight")
        rightFoot   = memoryProxy.getData("rightFootTotalWeight")
        print ("Total weight on left foot: %.2f kg, on right foot: %.2f kg" % (leftFoot, rightFoot))
        time.sleep(1.0)

    print("Foot contact lost")


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python sensors_fsr.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
