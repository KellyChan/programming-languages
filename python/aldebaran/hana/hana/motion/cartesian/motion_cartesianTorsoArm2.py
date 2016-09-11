# -*- encoding: UTF-8 -*- 

'''Cartesian control: Multiple Effector Trajectories'''

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
    ''' Move the torso and keep arms fixed in nao space
    Warning: Needs a PoseInit before executing
    '''

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

    space      = motion.FRAME_ROBOT
    isAbsolute = False

    effectorList = ["LArm", "RArm"]

    # Motion of Arms with block process
    axisMaskList = [almath.AXIS_MASK_VEL, almath.AXIS_MASK_VEL]

    timesList    = [[1.0], [1.0]]         # seconds
    pathList     = [[[0.0, -0.04, 0.0, 0.0, 0.0, 0.0]],
                    [[0.0,  0.04, 0.0, 0.0, 0.0, 0.0]]]
    motionProxy.positionInterpolations(effectorList, space, pathList,
                                       axisMaskList, timesList, isAbsolute)


    effectorList = ["LArm", "RArm", "Torso"]

    # Motion of Arms and Torso with block process
    axisMaskList = [almath.AXIS_MASK_VEL,
                    almath.AXIS_MASK_VEL,
                    almath.AXIS_MASK_ALL]

    timesList    = [[4.0],                  # LArm  in seconds
                    [4.0],                  # RArm  in seconds
                    [1.0, 2.0, 3.0, 4.0]]   # Torso in seconds

    dx           = 0.03                  # translation axis X (meters)
    dy           = 0.04                  # translation axis Y (meters)

    pathList     = [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], # LArm do not move
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], # RArm do not move
                    [[0.0, +dy, 0.0, 0.0, 0.0, 0.0],  # Torso point 1
                     [-dx, 0.0, 0.0, 0.0, 0.0, 0.0],  # Torso point 2
                     [0.0, -dy, 0.0, 0.0, 0.0, 0.0],  # Torso point 3
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # Torso point 4
                    ]
    motionProxy.positionInterpolations(effectorList, space, pathList,
                                       axisMaskList, timesList, isAbsolute)


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_cartesianTorsoArm2.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
