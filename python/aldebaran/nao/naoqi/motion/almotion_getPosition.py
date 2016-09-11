# -*- encoding: UTF-8 -*-

import motion
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    # Example showing how to get the position of the top camera
    name            = "CameraTop"
    frame           = motion.FRAME_WORLD
    useSensorValues = True
    result          = motionProxy.getPosition(name, frame, useSensorValues)
    print "Position of", name, " in World is:"
    print result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
