# -*- encoding: UTF-8 -*-

import motion
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    # Example showing how to get the COM position of "HeadYaw".
    name = "HeadYaw"
    frame = motion.FRAME_TORSO
    useSensors = False
    pos = motionProxy.getCOM(name, frame, useSensors)
    print "HeadYaw COM Position: x = ", pos[0], " y:", pos[1], " z:", pos[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
