# -*- encoding: UTF-8 -*-

import time
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    motionProxy.setStiffnesses("Head", 1.0)

    # Example simulating reactive control
    names = "HeadYaw"
    angles = 0.3
    fractionMaxSpeed = 0.1
    motionProxy.setAngles(names,angles,fractionMaxSpeed)
    # wait half a second
    time.sleep(0.5)
    # change target
    angles = 0.0
    motionProxy.setAngles(names,angles,fractionMaxSpeed)
    # wait half a second
    time.sleep(0.5)
    # change target
    angles = 0.1
    motionProxy.setAngles(names,angles,fractionMaxSpeed)

    time.sleep(3.0)
    motionProxy.setStiffnesses("Head", 0.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
