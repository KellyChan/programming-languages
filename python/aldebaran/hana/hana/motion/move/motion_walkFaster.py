# -*- encoding: UTF-8 -*- 

''' Walk: Small example to make Nao walk '''
'''       Faster (Step of 6cm)          '''

import sys
import time
from naoqi import ALProxy


def StiffnessOn(proxy):
    # We use the "Body" name to signify the collection of all joints
    pNames = "Body"
    pStiffnessLists = 1.0
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)


def main(robotIP):
    # Init proxies.
    try:
        motionProxy = ALProxy("ALMotion", robotIP, 9559)
    except Exception, e:
        print "Could not create proxy to ALMotion"
        print "Error was: ", e

    try:
        postureProxy = ALProxy("ALRobotPosture", robotIP, 9559)
    except Exception, e:
        print "Could not create proxy to ALRobotPosture"
        print "Error was: ", e

    # Set NAO in Stiffness On
    StiffnessOn(motionProxy)

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    # Initialize the walk process.
    # Check the robot pose and take a right posture.
    # This is blocking called.
    motionProxy.moveInit()

    # TARGET VELOCITY
    X         = 1.0
    Y         = 0.0
    Theta     = 0.0
    Frequency = 1.0

    # Default walk (MaxStepX = 0.04 m)
    motionProxy.setWalkTargetVelocity(X, Y, Theta, Frequency)
    time.sleep(3.0)
    print "walk Speed X :",motionProxy.getRobotVelocity()[0]," m/s"

    # Speed walk  (MaxStepX = 0.06 m)
    # Could be faster: see walk documentation
    motionProxy.setWalkTargetVelocity(X, Y, Theta, Frequency, [["MaxStepX", 0.06]])
    time.sleep(4.0)
    print "walk Speed X :",motionProxy.getRobotVelocity()[0]," m/s"

    # stop walk on the next double support
    motionProxy.stopMove()


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_walkFaster.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
