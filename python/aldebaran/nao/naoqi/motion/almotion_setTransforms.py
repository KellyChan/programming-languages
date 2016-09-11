# -*- encoding: UTF-8 -*-

import time
import motion
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy  = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)

    # Wake up robot
    motionProxy.wakeUp()

    # Send robot to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    # Example showing how to set Torso Transform, using a fraction of max speed
    chainName        = "Torso"
    frame            = motion.FRAME_ROBOT
    transform        = [1.0, 0.0, 0.0, 0.00,
                        0.0, 1.0, 0.0, 0.00,
                        0.0, 0.0, 1.0, 0.25]
    fractionMaxSpeed = 0.2
    axisMask         = 63
    motionProxy.setTransforms(chainName, frame, transform, fractionMaxSpeed, axisMask)

    time.sleep(4.0)

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
