# -*- encoding: UTF-8 -*-

'''Task management: the second motion is postponed'''

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
    angles = [0.0, 0.0]
    times  = [1.0, 1.0]
    isAbsolute = True
    motionProxy.angleInterpolation(names, angles, times, isAbsolute)

    # move slowly the head to look in the left direction. The motion will
    # take 4 seconds
    names  = "HeadYaw"
    angles = math.pi/2.0
    times  = 4.0
    isAbsolute = True
    motionProxy.post.angleInterpolation(names, angles, times, isAbsolute)

    time.sleep(1.0)

    # one second after having started motion1, check the resources use.
    # As expected the "HeadYaw" resource is busy
    isAvailable = motionProxy.areResourcesAvailable([names])
    print("areResourcesAvailable({0}): {1}".format([names], isAvailable))

    time.sleep(1.0)

    # try to look in the right direction. As the "HeadYaw" joint is busy,
    # this motion is postponed until the resource is freed
    angles = -math.pi/2.0
    times  = 2.0
    isAbsolute = True
    motionProxy.post.angleInterpolation(names, angles, times, isAbsolute)

    time.sleep(3.0)
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
