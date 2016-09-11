# -*- encoding: UTF-8 -*-

import almath
import time
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    motionProxy.setStiffnesses("Head", 1.0)

    # This function is useful to kill motion Task
    # To see the current motionTask please use getTaskList() function

    motionProxy.post.angleInterpolation('HeadYaw', 90*almath.TO_RAD, 10, True)
    time.sleep(3)
    taskList = motionProxy.getTaskList()
    uiMotion = taskList[0][1]
    motionProxy.killTask(uiMotion)

    motionProxy.setStiffnesses("Head", 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
