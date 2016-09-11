# -*- encoding: UTF-8 -*- 

'''Cartesian control: Arm trajectory example'''

import sys
import motion
from naoqi import ALProxy


def StiffnessOn(proxy):
    # We use the "Body" name to signify the collection of all joints
    pNames = "Body"
    pStiffnessLists = 1.0
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)


def main(robotIP):
    ''' Example showing a hand ellipsoid
    Warning: Needs a PoseInit before executing
    '''

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

    effector   = "LArm"
    space      = motion.FRAME_ROBOT
    path       = [
     [0.0, -0.05, +0.00, 0.0, 0.0, 0.0],        # point 1
     [0.0, +0.00, +0.04, 0.0, 0.0, 0.0],        # point 2
     [0.0, +0.04, +0.00, 0.0, 0.0, 0.0],        # point 3
     [0.0, +0.00, -0.02, 0.0, 0.0, 0.0],        # point 4
     [0.0, -0.05, +0.00, 0.0, 0.0, 0.0],        # point 5
     [0.0, +0.00, +0.00, 0.0, 0.0, 0.0]]        # point 6
    axisMask   = 7                              # just control position
    times      = [0.5, 1.0, 2.0, 3.0, 4.0, 4.5] # seconds
    isAbsolute = False
    motionProxy.positionInterpolation(effector, space, path,
                                      axisMask, times, isAbsolute)


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_cartesianArm2.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
