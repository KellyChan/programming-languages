# -*- encoding: UTF-8 -*-

import time
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy  = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)

    # Wake up robot
    motionProxy.wakeUp()

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    #####################################
    # A small example using getFootSteps
    #####################################

    # First call of move API
    # with post prefix to not be bloquing here.
    motionProxy.post.moveTo(0.3, 0.0, 0.5)

    # wait that the move process start running
    time.sleep(1.0)

    # get the foot steps vector
    footSteps = motionProxy.getFootSteps()

    # print the result
    leftFootWorldPosition = footSteps[0][0]
    print "leftFootWorldPosition:"
    print leftFootWorldPosition
    print ""

    rightFootWorldPosition = footSteps[0][1]
    print "rightFootWorldPosition:"
    print rightFootWorldPosition
    print ""

    footStepsUnchangeable = footSteps[1]
    print "Unchangeable:"
    print footStepsUnchangeable
    print ""

    footStepsChangeable   = footSteps[2]
    print "Changeable:"
    print footStepsChangeable
    print ""

    motionProxy.waitUntilMoveIsFinished()

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
