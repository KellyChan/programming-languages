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

    # Example showing the use of moveToward
    # The parameters are fractions of the maximums
    # Here we are asking for full speed forwards
    x     = 1.0
    y     = 0.0
    theta = 0.0
    frequency = 1.0
    motionProxy.moveToward(x, y, theta, [["Frequency", frequency]])

    # If we don't send another command, he will move forever
    # Lets make him slow down(step length) and turn after 3 seconds
    time.sleep(3)
    x     = 0.5
    theta = 0.6
    motionProxy.moveToward(x, y, theta, [["Frequency", frequency]])

    # Lets make him slow down(frequency) after 3 seconds
    time.sleep(3)
    frequency = 0.5
    motionProxy.moveToward(x, y, theta, [["Frequency", frequency]])

    # Lets make him stop after 3 seconds
    time.sleep(3)
    motionProxy.stopMove()
    # Note that at any time, you can use a moveTo command
    # to run a precise distance. The last command received,
    # of velocity or position always wins

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
