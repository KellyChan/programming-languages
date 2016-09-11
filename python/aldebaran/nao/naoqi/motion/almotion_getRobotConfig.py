# -*- encoding: UTF-8 -*-

import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    # Example showing how to get the robot config
    robotConfig = motionProxy.getRobotConfig()
    for i in range(len(robotConfig[0])):
        print robotConfig[0][i], ": ", robotConfig[1][i]

    # "Model Type"   : "naoH25", "naoH21", "naoT14" or "naoT2".
    # "Head Version" : "VERSION_32" or "VERSION_33" or "VERSION_40".
    # "Body Version" : "VERSION_32" or "VERSION_33" or "VERSION_40".
    # "Laser"        : True or False.
    # "Legs"         : True or False.
    # "Arms"         : True or False.
    # "Extended Arms": True or False.
    # "Hands"        : True or False.
    # "Arm Version"  : "VERSION_32" or "VERSION_33" or "VERSION_40".
    # Number of Legs : 0 or 2
    # Number of Arms : 0 or 2
    # Number of Hands: 0 or 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
