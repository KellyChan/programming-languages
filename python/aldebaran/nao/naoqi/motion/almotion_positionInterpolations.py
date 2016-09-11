# -*- encoding: UTF-8 -*-

import almath
import motion
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy  = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)

    # Wake up robot
    motionProxy.wakeUp()

    # Send robot to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    # Example showing how to use positionInterpolations
    frame = motion.FRAME_ROBOT
    useSensorValues = False

    dx = 0.03 # translation axis X (meters)
    dy = 0.04 # translation axis Y (meters)

    # Motion of Arms with block process
    effectorList = []
    pathList     = []

    axisMaskList = [motion.AXIS_MASK_VEL, motion.AXIS_MASK_VEL]
    timeList     = [[1.0], [1.0]]         # seconds

    effectorList.append("LArm")
    currentPos = motionProxy.getPosition("LArm", frame, useSensorValues)
    targetPos = almath.Position6D(currentPos)
    targetPos.y -= dy
    pathList.append(list(targetPos.toVector()))

    effectorList.append("RArm")
    currentPos = motionProxy.getPosition("RArm", frame, useSensorValues)
    targetPos = almath.Position6D(currentPos)
    targetPos.y += dy
    pathList.append(list(targetPos.toVector()))

    motionProxy.positionInterpolations(effectorList, frame, pathList,
                                 axisMaskList, timeList)

    # Motion of Arms and Torso with block process
    axisMaskList = [motion.AXIS_MASK_VEL,
                    motion.AXIS_MASK_VEL,
                    motion.AXIS_MASK_ALL]
    timeList    = [[4.0],
                    [4.0],
                    [1.0, 2.0, 3.0, 4.0]] # seconds

    effectorList = []
    pathList     = []

    effectorList.append("LArm")
    pathList.append([motionProxy.getPosition("LArm", frame, useSensorValues)])

    effectorList.append("RArm")
    pathList.append([motionProxy.getPosition("RArm", frame, useSensorValues)])

    effectorList.append("Torso")
    torsoList  = []
    currentPos = motionProxy.getPosition("Torso", frame, useSensorValues)
    targetPos  = almath.Position6D(currentPos)
    targetPos.y += dy
    torsoList.append(list(targetPos.toVector()))

    targetPos = almath.Position6D(currentPos)
    targetPos.x -= dx
    torsoList.append(list(targetPos.toVector()))

    targetPos = almath.Position6D(currentPos)
    targetPos.y -= dy
    torsoList.append(list(targetPos.toVector()))

    targetPos = almath.Position6D(currentPos)
    torsoList.append(list(targetPos.toVector()))
    pathList.append(torsoList)

    motionProxy.positionInterpolations(effectorList, frame, pathList,
                                 axisMaskList, timeList)

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
