# -*- encoding: UTF-8 -*- 

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
    '''
         Example showing a Hula Hoop Motion
         with the NAO cartesian control of torso
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

    # Define the changes relative to the current position
    dx         = 0.07                    # translation axis X (meter)
    dy         = 0.07                    # translation axis Y (meter)
    dwx        = 0.15                    # rotation axis X (rad)
    dwy        = 0.15                    # rotation axis Y (rad)

    # Define a path of two hula hoop loops
    path = [ [+dx, 0.0, 0.0,  0.0, -dwy, 0.0],  # point 01 : forward  / bend backward
             [0.0, -dy, 0.0, -dwx,  0.0, 0.0],  # point 02 : right    / bend left
             [-dx, 0.0, 0.0,  0.0,  dwy, 0.0],  # point 03 : backward / bend forward
             [0.0, +dy, 0.0,  dwx,  0.0, 0.0],  # point 04 : left     / bend right

             [+dx, 0.0, 0.0,  0.0, -dwy, 0.0],  # point 01 : forward  / bend backward
             [0.0, -dy, 0.0, -dwx,  0.0, 0.0],  # point 02 : right    / bend left
             [-dx, 0.0, 0.0,  0.0,  dwy, 0.0],  # point 03 : backward / bend forward
             [0.0, +dy, 0.0,  dwx,  0.0, 0.0],  # point 04 : left     / bend right

             [+dx, 0.0, 0.0,  0.0, -dwy, 0.0],  # point 05 : forward  / bend backward
             [0.0, 0.0, 0.0,  0.0,  0.0, 0.0] ] # point 06 : Back to init pose

    timeOneMove  = 0.4 #seconds
    times = []
    for i in range(len(path)):
        times.append( (i+1)*timeOneMove )

    # call the cartesian control API
    effector   = "Torso"
    space      =  motion.FRAME_ROBOT

    axisMask   = almath.AXIS_MASK_ALL
    isAbsolute = False

    motionProxy.positionInterpolation(effector, space, path,
                                      axisMask, times, isAbsolute)


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_hulaHoop.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
