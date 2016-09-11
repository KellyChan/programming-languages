# -*- encoding: UTF-8 -*-

import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    # Example showing how to set the stiffness to 1.0.
    # Beware, doing this could be dangerous, it is safer to use the
    #   stiffnessInterpolation method which takes a duration parameter.
    names  = 'Body'
    # If only one parameter is received, this is applied to all joints
    stiffnesses  = 1.0
    motionProxy.setStiffnesses(names, stiffnesses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
