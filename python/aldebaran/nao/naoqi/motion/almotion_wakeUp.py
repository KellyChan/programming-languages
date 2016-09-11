# -*- encoding: UTF-8 -*-

'''Wake up: sets Motor on and, if needed, goes to initial position'''

import time
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):

    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    motionProxy.wakeUp()

    # print motion state
    print motionProxy.getSummary()
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
