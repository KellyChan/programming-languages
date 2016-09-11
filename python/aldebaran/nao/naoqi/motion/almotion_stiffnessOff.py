# -*- encoding: UTF-8 -*-

'''Stiffness Off: remove stiffness of all joints and actuators'''
''' This example is only compatible with NAO '''

import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):

    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    # We use the "Body" name to signify the collection of all joints
    names = "Body"
    stiffnessLists = 0.0
    timeLists = 1.0
    motionProxy.stiffnessInterpolation(names, stiffnessLists, timeLists)

    # print motion state
    print motionProxy.getSummary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
