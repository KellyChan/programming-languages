# -*- encoding: UTF-8 -*- 

'''Cartesian control: Torso and Foot trajectories'''

import sys
import motion
import almath
from naoqi import ALProxy


def StiffnessOn(proxy):
    # We use the "Body" name to signify the collection of all joints
    pNames = "Body"
    pStiffnessLists = 1.0
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)


def main(robotIP):
    ''' Example of a cartesian foot trajectory
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

    space      = motion.FRAME_ROBOT
    axisMask   = almath.AXIS_MASK_ALL   # full control
    isAbsolute = False

    # Lower the Torso and move to the side
    effector   = "Torso"
    path       = [0.0, -0.07, -0.03, 0.0, 0.0, 0.0]
    time       = 2.0                    # seconds
    motionProxy.positionInterpolation(effector, space, path,
                                axisMask, time, isAbsolute)

    # LLeg motion
    effector   = "LLeg"
    path       = [0.0,  0.06,  0.00, 0.0, 0.0, 0.8]
    times      = 2.0                    # seconds

    motionProxy.positionInterpolation(effector, space, path,
                                      axisMask, times, isAbsolute)


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_cartesianFoot.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
