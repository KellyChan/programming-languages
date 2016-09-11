# -*- encoding: UTF-8 -*-

import almath
import time
import argparse
from naoqi import ALProxy

def main(robotIP, PORT = 9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    motionProxy.setStiffnesses("Head", 1.0)

    # Example showing a single target angle for one joint
    # Interpolates the head yaw to 1.0 radian in 1.0 second
    names      = ["HeadYaw"]
    angleLists = [[50.0*almath.TO_RAD]]
    timeLists  = [[1.0]]
    motionProxy.angleInterpolationBezier(names, timeLists, angleLists)

    time.sleep(1.0)

    # Example showing a single trajectory for one joint
    # Interpolates the head yaw to 1.0 radian and back to zero in 2.0 seconds
    names      = ["HeadYaw"]
    #              2 angles
    angleLists = [[30.0*almath.TO_RAD, 0.0]]
    #              2 times
    timeLists  = [[1.0, 2.0]]
    motionProxy.angleInterpolationBezier(names, timeLists, angleLists)

    time.sleep(1.0)

    # Example showing multiple trajectories
    names      = ["HeadYaw", "HeadPitch"]
    angleLists = [[30.0*almath.TO_RAD], [30.0*almath.TO_RAD]]
    timeLists  = [[1.0], [1.2]]
    motionProxy.angleInterpolationBezier(names, timeLists, angleLists)

    # Example showing multiple trajectories
    # Interpolates the head yaw to 1.0 radian and back to zero in 2.0 seconds
    # while interpolating HeadPitch up and down over a longer period.
    names  = ["Head"]
    # Each joint can have lists of different lengths, but the number of
    # angles and the number of times must be the same for each joint.
    # Here, the second joint ("HeadPitch") has three angles, and
    # three corresponding times.
    angleLists  = [[50.0*almath.TO_RAD, 0.0],
                   [-30.0*almath.TO_RAD, 30.0*almath.TO_RAD, 0.0]]
    timeLists   = [[1.0, 2.0], [ 1.0, 2.0, 3.0]]
    motionProxy.angleInterpolationBezier(names, timeLists, angleLists)

    motionProxy.setStiffnesses("Head", 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
