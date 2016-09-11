# -*- encoding: UTF-8 -*- 

'''Cartesian control: Torso trajectory'''

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
    ''' Example showing a path of five positions
    Needs a PoseInit before execution
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

    effector   = "Torso"
    space      =  motion.FRAME_WORLD

    # Position Only
    axisMask   = almath.AXIS_MASK_ALL           # full control
    isAbsolute = False

    # Define the changes relative to the current position
    dx         = 0.045                          # translation axis X (meter)
    dy         = 0.050                          # translation axis Y (meter)

    path       = [
      [+dx, 0.0, 0.0, 0.0, 0.0, 0.0],           # point 1
      [0.0, -dy, 0.0, 0.0, 0.0, 0.0],           # point 2
      [-dx, 0.0, 0.0, 0.0, 0.0, 0.0],           # point 3
      [0.0, +dy, 0.0, 0.0, 0.0, 0.0],           # point 4
      [+dx, 0.0, 0.0, 0.0, 0.0, 0.0],           # point 5
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]           # point 6

    times      = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] # seconds
    motionProxy.positionInterpolation(effector, space, path,
                                axisMask, times, isAbsolute)


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_cartesianTorso.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
