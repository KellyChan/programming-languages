# -*- encoding: UTF-8 -*-

import almath
import time
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    motionProxy.setStiffnesses("Head", 1.0)

    # Head Start to zeros
    names             = "Head"
    targetAngles      = [0.0, 0.0]
    maxSpeedFraction  = 0.2 # Using 20% of maximum joint speed
    motionProxy.angleInterpolationWithSpeed(names, targetAngles, maxSpeedFraction)

    # Example showing a reactive control with time function angleInterpolation
    # Goal: after 1.0 second, retarget from 40 to 0 degree: smooth transition

    # Interpolate the head yaw to 20 degrees in 2.0 seconds
    # With post, angleInterpolation become non-blocking
    names      = "HeadYaw"
    angleLists = 20.0*almath.TO_RAD
    timeLists  = 2.0
    isAbsolute = True
    motionProxy.post.angleInterpolation(names, angleLists, timeLists, isAbsolute)
    time.sleep(0.5)

    # Call getTaskList to have the previous angleInterpolation task number
    taskList = motionProxy.getTaskList()

    # Prepare the next target to 50.0 degrees in 1.0 second
    angleLists = 50.0*almath.TO_RAD
    timeLists  = 1.0
    motionProxy.post.angleInterpolation(names, angleLists, timeLists, isAbsolute)
    time.sleep(0.5)

    # Kill the first angleInterpolation (go to 40.0 degrees), the second start
    # smoothly from the current joint position and velocity (go to 0 degrees)
    motionProxy.killTask(taskList[0][1])

    time.sleep(2.0)
    motionProxy.setStiffnesses("Head", 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
