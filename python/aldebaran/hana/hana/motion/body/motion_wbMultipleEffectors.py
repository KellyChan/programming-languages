# -*- encoding: UTF-8 -*- 

''' Whole Body Motion: Multiple Effectors control '''

import sys
import motion
import almath
import time
from naoqi import ALProxy


def StiffnessOn(proxy):
    # We use the "Body" name to signify the collection of all joints
    pNames = "Body"
    pStiffnessLists = 1.0
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)


def main(robotIP):
    '''
        Example of a whole body multiple effectors control "LArm", "RArm" and "Torso"
        Warning: Needs a PoseInit before executing
                 Whole body balancer must be inactivated at the end of the script
    '''

    # Init proxies.
    try:
        proxy = ALProxy("ALMotion", robotIP, 9559)
    except Exception, e:
        print "Could not create proxy to ALMotion"
        print "Error was: ", e

    try:
        postureProxy = ALProxy("ALRobotPosture", robotIP, 9559)
    except Exception, e:
        print "Could not create proxy to ALRobotPosture"
        print "Error was: ", e

    # Set NAO in Stiffness On
    StiffnessOn(proxy)

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    # Enable Whole Body Balancer
    isEnabled  = True
    proxy.wbEnable(isEnabled)

    # Legs are constrained fixed
    stateName  = "Fixed"
    supportLeg = "Legs"
    proxy.wbFootState(stateName, supportLeg)

    # Constraint Balance Motion
    isEnable   = True
    supportLeg = "Legs"
    proxy.wbEnableBalanceConstraint(isEnable, supportLeg)

    # Arms motion
    effectorList = ["LArm", "RArm"]

    space        = motion.FRAME_ROBOT

    pathList     = [
                    [
                     [0.0,   0.08,  0.14, 0.0, 0.0, 0.0], # target 1 for "LArm"
                     [0.0,  -0.05, -0.07, 0.0, 0.0, 0.0], # target 2 for "LArm"
                     [0.0,   0.08,  0.14, 0.0, 0.0, 0.0], # target 3 for "LArm"
                     [0.0,  -0.05, -0.07, 0.0, 0.0, 0.0], # target 4 for "LArm"
                     [0.0,   0.08,  0.14, 0.0, 0.0, 0.0], # target 5 for "LArm"
                     ],
                    [
                     [0.0,   0.05, -0.07, 0.0, 0.0, 0.0], # target 1 for "RArm"
                     [0.0,  -0.08,  0.14, 0.0, 0.0, 0.0], # target 2 for "RArm"
                     [0.0,   0.05, -0.07, 0.0, 0.0, 0.0], # target 3 for "RArm"
                     [0.0,  -0.08,  0.14, 0.0, 0.0, 0.0], # target 4 for "RArm"
                     [0.0,   0.05, -0.07, 0.0, 0.0, 0.0], # target 5 for "RArm"
                     [0.0,  -0.08,  0.14, 0.0, 0.0, 0.0], # target 6 for "RArm"
                     ]
                    ]

    axisMaskList = [almath.AXIS_MASK_VEL, # for "LArm"
                    almath.AXIS_MASK_VEL] # for "RArm"

    coef       = 1.5
    timesList  = [ [coef*(i+1) for i in range(5)],  # for "LArm" in seconds
                   [coef*(i+1) for i in range(6)] ] # for "RArm" in seconds

    isAbsolute   = False

    # called cartesian interpolation
    proxy.positionInterpolations(effectorList, space, pathList,
                                 axisMaskList, timesList, isAbsolute)

    # Torso Motion
    effectorList = ["Torso", "LArm", "RArm"]

    dy = 0.06
    dz = 0.06
    pathList     = [
                    [
                     [0.0, +dy, -dz, 0.0, 0.0, 0.0], # target  1 for "Torso"
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # target  2 for "Torso"
                     [0.0, -dy, -dz, 0.0, 0.0, 0.0], # target  3 for "Torso"
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # target  4 for "Torso"
                     [0.0, +dy, -dz, 0.0, 0.0, 0.0], # target  5 for "Torso"
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # target  6 for "Torso"
                     [0.0, -dy, -dz, 0.0, 0.0, 0.0], # target  7 for "Torso"
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # target  8 for "Torso"
                     [0.0, +dy, -dz, 0.0, 0.0, 0.0], # target  9 for "Torso"
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # target 10 for "Torso"
                     [0.0, -dy, -dz, 0.0, 0.0, 0.0], # target 11 for "Torso"
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # target 12 for "Torso"
                     ],
                     [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], # for "LArm"
                     [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], # for "LArm"
                    ]

    axisMaskList = [almath.AXIS_MASK_ALL, # for "Torso"
                    almath.AXIS_MASK_VEL, # for "LArm"
                    almath.AXIS_MASK_VEL] # for "RArm"

    coef       = 0.5
    timesList  = [
                  [coef*(i+1) for i in range(12)], # for "Torso" in seconds
                  [coef*12],                       # for "LArm" in seconds
                  [coef*12]                        # for "RArm" in seconds
                 ]

    isAbsolute   = False

    proxy.positionInterpolations(effectorList, space, pathList,
                                 axisMaskList, timesList, isAbsolute)


    # Deactivate whole body
    isEnabled    = False
    proxy.wbEnable(isEnabled)

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_wbMultipleEffectors.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
