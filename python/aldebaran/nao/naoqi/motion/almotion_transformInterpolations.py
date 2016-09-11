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

    # Example showing how to use transformInterpolations
    frame        = motion.FRAME_ROBOT
    isAbsolute   = False
    useSensorValues = False

    # Motion of Arms with block process
    effectorList = ["LArm", "RArm"]
    axisMaskList = [motion.AXIS_MASK_VEL, motion.AXIS_MASK_VEL]
    timeList     = [[1.0], [1.0]]         # seconds

    dy = 0.04
    pathList = []
    targetLArmTf = almath.Transform(motionProxy.getTransform("LArm", frame, useSensorValues))
    targetLArmTf.r2_c4 -= dy
    pathList.append(list(targetLArmTf.toVector()))

    targetRArmTf = almath.Transform(motionProxy.getTransform("RArm", frame, useSensorValues))
    targetRArmTf.r2_c4 += dy
    pathList.append(list(targetRArmTf.toVector()))

    motionProxy.transformInterpolations(effectorList, frame, pathList,
                                 axisMaskList, timeList)

    # Motion of Arms and Torso with block process
    effectorList = ["LArm", "RArm", "Torso"]
    axisMaskList = [motion.AXIS_MASK_VEL,
                    motion.AXIS_MASK_VEL,
                    motion.AXIS_MASK_ALL]
    timeList     = [[4.0],
                    [4.0],
                    [1.0, 2.0, 3.0, 4.0]] # seconds

    dx = 0.03 # translation axis X (meters)
    dy = 0.05 # translation axis Y (meters)

    pathList = []

    targetLArmTf = almath.Transform(motionProxy.getTransform("LArm", frame, useSensorValues))
    pathList.append(list(targetLArmTf.toVector()))

    targetRArmTf = almath.Transform(motionProxy.getTransform("RArm", frame, useSensorValues))
    pathList.append(list(targetRArmTf.toVector()))

    pathTorsoList = []
    # point 1
    initTorsoTf = almath.Transform(motionProxy.getTransform("Torso", frame, useSensorValues))
    targetTorsoTf = initTorsoTf
    targetTorsoTf.r2_c4 += dy
    pathTorsoList.append(list(targetTorsoTf.toVector()))

    # point 2
    initTorsoTf = almath.Transform(motionProxy.getTransform("Torso", frame, useSensorValues))
    targetTorsoTf = initTorsoTf
    targetTorsoTf.r1_c4 += -dx
    pathTorsoList.append(list(targetTorsoTf.toVector()))

    # point 3
    initTorsoTf = almath.Transform(motionProxy.getTransform("Torso", frame, useSensorValues))
    targetTorsoTf = initTorsoTf
    targetTorsoTf.r2_c4 += -dy
    pathTorsoList.append(list(targetTorsoTf.toVector()))

    # point 4
    initTorsoTf = almath.Transform(motionProxy.getTransform("Torso", frame, useSensorValues))
    pathTorsoList.append(list(initTorsoTf.toVector()))

    pathList.append(pathTorsoList)

    motionProxy.transformInterpolations(effectorList, frame, pathList,
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
