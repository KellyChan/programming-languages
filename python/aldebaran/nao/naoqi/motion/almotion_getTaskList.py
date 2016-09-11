# -*- encoding: UTF-8 -*-

import time
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    motionProxy.setStiffnesses("Head", 1.0)

    # Example showing how to get the current task list
    # We will create a task first, so that we see a result
    names      = "HeadYaw"
    angleLists = 1.0
    timeList   = 3.0
    isAbsolute = True
    motionProxy.post.angleInterpolation(names, angleLists, timeList, isAbsolute)
    time.sleep(0.1)
    print 'Tasklist: ', motionProxy.getTaskList()

    time.sleep(2.0)

    motionProxy.setStiffnesses("Head", 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
