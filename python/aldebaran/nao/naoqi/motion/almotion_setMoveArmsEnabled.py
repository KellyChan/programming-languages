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

    x          = 0.6
    y          = 0.0
    theta      = 0.0
    frequency  = 1.0
    motionProxy.moveToward(x, y, theta, [["Frequency", frequency]])

    time.sleep(2.0)

    # Example showing how to disable left arm motions during a move
    leftArmEnable  = False
    rightArmEnable = True
    motionProxy.setMoveArmsEnabled(leftArmEnable, rightArmEnable)
    print "Disabled left arm"

    # disable also right arm motion after 1 seconds
    time.sleep(1.0)
    rightArmEnable  = False
    motionProxy.setMoveArmsEnabled(leftArmEnable, rightArmEnable)
    print "Disabled right arm"

    time.sleep(1.0)

    motionProxy.stopMove()

    leftArmEnable  = True
    rightArmEnable = True
    motionProxy.setMoveArmsEnabled(leftArmEnable, rightArmEnable)

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
