# -*- encoding: UTF-8 -*-

import time
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy  = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)

    # Wake up robot
    motionProxy.wakeUp()

    # Send robot to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    # A small step forwards and anti-clockwise with the left foot
    legName = ["LLeg"]
    X       = 0.04
    Y       = 0.1
    Theta   = 0.3
    footSteps = [[X, Y, Theta]]
    fractionMaxSpeed = [1.0]
    clearExisting = False
    motionProxy.setFootStepsWithSpeed(legName, footSteps, fractionMaxSpeed, clearExisting)

    time.sleep(0.5)

    # A small step forwards and anti-clockwise with the left foot
    legName = ["LLeg", "RLeg"]
    X       = 0.04
    Y       = 0.1
    Theta   = 0.3
    footSteps = [[X, Y, Theta], [X, -Y, Theta]]
    fractionMaxSpeed = [1.0, 1.0]
    clearExisting = False
    motionProxy.setFootStepsWithSpeed(legName, footSteps, fractionMaxSpeed, clearExisting)

    motionProxy.waitUntilMoveIsFinished()

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
