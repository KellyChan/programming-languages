# -*- encoding: UTF-8 -*- 

''' Walk: Small example to make Nao walk '''
'''       with jerky head                '''

import sys
import time
import random
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
    postureProxy.goToPosture("StandInit", 1.0)

    # Initialize the walk process.
    # Check the robot pose and take a right posture.
    # This is blocking called.
    motionProxy.moveInit()

    testTime = 10 # seconds
    t = 0
    dt = 0.2
    while (t<testTime):

        # WALK
        X         = random.uniform(0.4, 1.0)
        Y         = random.uniform(-0.4, 0.4)
        Theta     = random.uniform(-0.4, 0.4)
        Frequency = random.uniform(0.5, 1.0)
        motionProxy.setWalkTargetVelocity(X, Y, Theta, Frequency)

        # JERKY HEAD
        motionProxy.setAngles("HeadYaw", random.uniform(-1.0, 1.0), 0.6)
        motionProxy.setAngles("HeadPitch", random.uniform(-0.5, 0.5), 0.6)

        t = t + dt
        time.sleep(dt)

    # stop walk on the next double support
    motionProxy.stopMove()


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_walkWithJerkyHead.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
