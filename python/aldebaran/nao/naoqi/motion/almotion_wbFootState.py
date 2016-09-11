# -*- encoding: UTF-8 -*-

''' Whole Body Motion: Foot State '''
''' This example is only compatible with NAO '''

import argparse
import math
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    ''' Example of a whole body FootState
    Warning: Needs a PoseInit before executing
             Whole body balancer must be inactivated at the end of the script
    '''

    motionProxy  = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)

    # Wake up robot
    motionProxy.wakeUp()

    # Send robot to Stand Init
    postureProxy.goToPosture("StandInit", 0.5)

    # Activate Whole Body Balancer.
    isEnabled  = True
    motionProxy.wbEnable(isEnabled)

    # Legs are constrained in a plane
    stateName  = "Plane"
    supportLeg = "Legs"
    motionProxy.wbFootState(stateName, supportLeg)

    # HipYawPitch angleInterpolation
    # Without Whole Body balancer, foot will not be keeped plane.
    names      = "LHipYawPitch"
    angleLists = [-45.0, 10.0, 0.0]
    timeLists  = [3.0, 6.0, 9.0]
    isAbsolute = True
    angleLists = [angle*math.pi/180.0 for angle in angleLists]
    try:
        motionProxy.angleInterpolation(names, angleLists, timeLists, isAbsolute)
    except Exception, errorMsg:
        print str(errorMsg)
        print "This example is not allowed on this robot."
        exit()

    # Deactivate Whole Body Balancer.
    isEnabled  = False
    motionProxy.wbEnable(isEnabled)

    # Go to rest position
    motionProxy.rest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
