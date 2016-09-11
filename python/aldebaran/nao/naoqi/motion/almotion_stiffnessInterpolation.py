# -*- encoding: UTF-8 -*-

import time
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    # Example showing how to interpolate to maximum stiffness in 1 second
    names  = 'Body'
    stiffnessLists  = 0.0
    timeLists  = 1.0
    motionProxy.stiffnessInterpolation(names, stiffnessLists, timeLists)

    time.sleep(1.0)

    # Example showing a stiffness trajectory for a single joint
    names  = ['HeadYaw']
    stiffnessLists  = [0.25, 0.5, 1.0, 0.0]
    timeLists  = [1.0, 2.0, 3.0, 4.0]
    motionProxy.stiffnessInterpolation(names, stiffnessLists, timeLists)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
