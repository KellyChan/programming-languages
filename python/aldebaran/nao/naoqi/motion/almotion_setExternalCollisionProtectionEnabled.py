# -*- encoding: UTF-8 -*-

import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    # Example showing how to activate "Move", "LArm" and "RArm" external anti collision
    name = "All"
    enable  = True
    motionProxy.setExternalCollisionProtectionEnabled(name, enable)

    # Example showing how to deactivate "LArm" external anti collision
    name = "LArm"
    enable = False
    motionProxy.setExternalCollisionProtectionEnabled(name, enable)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
