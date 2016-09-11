# -*- encoding: UTF-8 -*-

''' Task management: the second motion is not postponed '''

import argparse
import math
import time
from naoqi import ALProxy

def main(robotIP, PORT=9559):

    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    # Wake up robot
    motionProxy.wakeUp()

    # go to an init head pose.
    names  = ["HeadYaw", "HeadPitch"]
    angles = [0., 0.]
    times  = [1.0, 1.0]
    isAbsolute = True
    motionProxy.angleInterpolation(names, angles, times, isAbsolute)

    # move slowly the head to look in the left direction
    names  = "HeadYaw"
    angles = math.pi/2
    fractionMaxSpeed = .1
    motionProxy.setAngles(names, angles, fractionMaxSpeed)

    time.sleep(1.)

    # while the previous motion is still running, update the angle
    angles  = -math.pi/6
    fractionMaxSpeed  = 1.
    motionProxy.setAngles(names, angles, fractionMaxSpeed)

    time.sleep(2.0)
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
