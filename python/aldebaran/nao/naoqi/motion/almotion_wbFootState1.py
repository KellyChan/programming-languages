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

    motionProxy.wbEnable(True)

    # Example showing how to fix the feet.
    print "Feet fixed."
    stateName = "Fixed"
    supportLeg = "Legs"
    motionProxy.wbFootState(stateName, supportLeg)

    # Example showing how to fix the left leg and constrained in a plane the right leg.
    print "Left leg fixed, right leg in a plane."
    motionProxy.wbFootState("Fixed", "LLeg")
    motionProxy.wbFootState("Plane", "RLeg")

    # Example showing how to fix the left leg and keep free the right leg.
    print "Left leg fixed, right leg free"
    motionProxy.wbFootState("Fixed", "LLeg")
    motionProxy.wbFootState("Free", "RLeg")

    time.sleep(2.0)
    motionProxy.wbEnable(False)

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
