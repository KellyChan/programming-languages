# -*- encoding: UTF-8 -*- 

''' Walk: Small example to make Nao walk '''
'''       with gait customization        '''
'''       NAO is Keyser Soze             '''

import sys
import time
import almath
from naoqi import ALProxy


def StiffnessOn(proxy):
    # We use the "Body" name to signify the collection of all joints
    pNames = "Body"
    pStiffnessLists = 1.0
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)


def main(robotIP = "127.0.0.1"):
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

    # TARGET VELOCITY
    X         = 1.0
    Y         = 0.0
    Theta     = 0.0
    Frequency = 1.0

    # Defined a limp walk
    motionProxy.setWalkTargetVelocity(X, Y, Theta, Frequency,
        [ # LEFT FOOT
        ["StepHeight", 0.02],
        ["TorsoWy", 5.0*almath.TO_RAD] ],
        [ # RIGHT FOOT
        ["StepHeight", 0.005],
        ["MaxStepX", 0.001],
        ["MaxStepFrequency", 0.0],
        ["TorsoWx", -7.0*almath.TO_RAD],
        ["TorsoWy", 5.0*almath.TO_RAD] ] )

    time.sleep(4.0)

    motionProxy.setWalkTargetVelocity(X, Y, Theta, Frequency)

    time.sleep(4.0)

    # stop walk in the next double support
    motionProxy.stopMove()


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_walkCustomization.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]
    main(robotIp)
