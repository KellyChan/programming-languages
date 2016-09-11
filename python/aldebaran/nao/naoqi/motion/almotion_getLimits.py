# -*- encoding: UTF-8 -*-

import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    # Example showing how to get the limits for the whole body
    name = "Body"
    limits = motionProxy.getLimits(name)
    jointNames = motionProxy.getBodyNames(name)
    for i in range(0,len(limits)):
        print jointNames[i] + ":"
        print "minAngle", limits[i][0],\
            "maxAngle", limits[i][1],\
            "maxVelocity", limits[i][2],\
            "maxTorque", limits[i][3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
