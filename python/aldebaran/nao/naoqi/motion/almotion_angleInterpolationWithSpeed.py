# -*- encoding: UTF-8 -*-

import time
import argparse
from naoqi import ALProxy

def main(robotIP, PORT = 9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)

    # Wake up robot
    motionProxy.wakeUp()

    # Send robot to Stand Init
    postureProxy.goToPosture("StandInit", 0.5)

    # Example showing multiple trajectories
    # Interpolate the head yaw to 1.0 radian and back to zero in 2.0 seconds
    # while interpolating HeadPitch up and down over a longer period.
    names  = ["HeadYaw","HeadPitch"]
    # Each joint can have lists of different lengths, but the number of
    # angles and the number of times must be the same for each joint.
    # Here, the second joint ("HeadPitch") has three angles, and
    # three corresponding times.
    angleLists  = [[1.0, 0.0], [-0.5, 0.5, 0.0]]
    timeLists   = [[1.0, 2.0], [ 1.0, 2.0, 3.0]]
    isAbsolute  = True
    motionProxy.angleInterpolation(names, angleLists, timeLists, isAbsolute)

    time.sleep(1.0)

    # Example showing a single target for one joint
    names             = "HeadYaw"
    targetAngles      = 1.0
    maxSpeedFraction  = 0.2 # Using 20% of maximum joint speed
    motionProxy.angleInterpolationWithSpeed(names, targetAngles, maxSpeedFraction)

    time.sleep(1.0)

    # Example showing multiple joints
    # Instead of listing each joint, you can use a chain name, which will
    # be expanded to contain all the joints in the chain. In this case,
    # "Head" will be interpreted as ["HeadYaw", "HeadPitch"]
    names  = "Head"
    # We still need to specify the correct number of target angles
    targetAngles     = [0.5, 0.25]
    maxSpeedFraction = 0.2 # Using 20% of maximum joint speed
    motionProxy.angleInterpolationWithSpeed(names, targetAngles, maxSpeedFraction)

    # Example showing body zero position
    # Instead of listing each joint, you can use a the name "Body"
    names  = "Body"
    # We still need to specify the correct number of target angles, so
    # we need to find the number of joints that this Nao has.
    # Here we are using the getBodyNames method, which tells us all
    # the names of the joints in the alias "Body".
    # We could have used this list for the "names" parameter.
    numJoints = len(motionProxy.getBodyNames("Body"))
    # Make a list of the correct length. All angles are zero.
    targetAngles  = [0.0]*numJoints
    # Using 10% of maximum joint speed
    maxSpeedFraction  = 0.1
    motionProxy.angleInterpolationWithSpeed(names, targetAngles, maxSpeedFraction)

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
