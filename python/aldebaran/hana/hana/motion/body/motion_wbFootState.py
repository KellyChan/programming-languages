# -*- encoding: UTF-8 -*- 

''' Whole Body Motion: Foot State '''

import sys
import math
from naoqi import ALProxy


def StiffnessOn(proxy):
    # We use the "Body" name to signify the collection of all joints
    pNames = "Body"
    pStiffnessLists = 1.0
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)


def main(robotIP):
    ''' Example of a whole body FootState
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

    # Activate Whole Body Balancer.
    isEnabled  = True
    proxy.wbEnable(isEnabled)

    # Legs are constrained in a plane
    stateName  = "Plane"
    supportLeg = "Legs"
    proxy.wbFootState(stateName, supportLeg)

    # HipYawPitch angleInterpolation
    # Without Whole Body balancer, foot will not be keeped plane.
    names      = "LHipYawPitch"
    angleLists = [-45.0, 10.0, 0.0]
    timeLists  = [3.0, 6.0, 9.0]
    isAbsolute = True
    angleLists = [angle*math.pi/180.0 for angle in angleLists]
    proxy.angleInterpolation(names, angleLists, timeLists, isAbsolute)

    # Deactivate Whole Body Balancer.
    isEnabled  = False
    proxy.wbEnable(isEnabled)


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_wbFootState.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
