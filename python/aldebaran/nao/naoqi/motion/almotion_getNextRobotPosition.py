# -*- encoding: UTF-8 -*-

import almath
import time
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy  = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)

    # Wake up robot
    motionProxy.wakeUp()

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    # Example showing how to get a simplified robot position in world.
    result = motionProxy.getNextRobotPosition()
    print "Next Robot Position", result

    # Example showing how to use this information to know the robot's diplacement
    # during the move process.

    # Make the robot move
    motionProxy.post.moveTo(0.6, 0.0, 0.5) # No blocking due to post called
    time.sleep(1.0)
    initRobotPosition = almath.Pose2D(motionProxy.getNextRobotPosition())

    # Make the robot move
    motionProxy.moveTo(0.1, 0.0, 0.2)

    endRobotPosition = almath.Pose2D(motionProxy.getNextRobotPosition())

    # Compute robot's' displacement
    robotMove = almath.pose2DInverse(initRobotPosition)*endRobotPosition
    print "Robot Move :", robotMove

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
