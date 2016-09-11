# -*- encoding: UTF-8 -*-

'''Cartesian control: Multiple Effector Trajectories'''
''' This example is only compatible with NAO '''

import argparse
import motion
import almath
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    ''' Simultaneously control three effectors:
    the Torso, the Left Arm and the Right Arm
    Warning: Needs a PoseInit before executing
    '''

    motionProxy  = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)

    # Wake up robot
    motionProxy.wakeUp()

    # Send robot to Stand Init
    postureProxy.goToPosture("StandInit", 0.5)

    frame      = motion.FRAME_WORLD
    coef       = 0.5                   # motion speed
    times      = [coef, 2.0*coef, 3.0*coef, 4.0*coef]
    useSensorValues = False

    # Relative movement between current and desired positions
    dy         = +0.03                 # translation axis Y (meters)
    dz         = -0.03                 # translation axis Z (meters)
    dwx        = +8.0*almath.TO_RAD   # rotation axis X (radians)

    # Motion of Torso with post process
    effector   = "Torso"

    path = []
    initTf = almath.Transform(motionProxy.getTransform(effector, frame, useSensorValues))
    # point 1
    deltaTf  = almath.Transform(0.0, -dy, dz)*almath.Transform().fromRotX(-dwx)
    targetTf = initTf*deltaTf
    path.append(list(targetTf.toVector()))

    # point 2
    path.append(list(initTf.toVector()))

    # point 3
    deltaTf  = almath.Transform(0.0, dy, dz)*almath.Transform().fromRotX(dwx)
    targetTf = initTf*deltaTf
    path.append(list(targetTf.toVector()))

    # point 4
    path.append(list(initTf.toVector()))

    axisMask   = almath.AXIS_MASK_ALL  # control all the effector axes
    motionProxy.post.transformInterpolations(effector, frame, path,
                                           axisMask, times)

    # Motion of Arms with block process
    frame     = motion.FRAME_TORSO
    axisMask  = almath.AXIS_MASK_VEL  # control just the position
    times     = [1.0*coef, 2.0*coef]  # seconds

    # Motion of Right Arm during the first half of the Torso motion
    effector  = "RArm"

    path = []
    currentTf = motionProxy.getTransform(effector, frame, useSensorValues)
    targetTf  = almath.Transform(currentTf)
    targetTf.r2_c4 -= 0.04 # y
    path.append(list(targetTf.toVector()))
    path.append(currentTf)

    motionProxy.transformInterpolations(effector, frame, path, axisMask, times)

    # Motion of Left Arm during the last half of the Torso motion
    effector   = "LArm"

    path = []
    currentTf = motionProxy.getTransform(effector, frame, useSensorValues)
    targetTf  = almath.Transform(currentTf)
    targetTf.r2_c4 += 0.04 # y
    path.append(list(targetTf.toVector()))
    path.append(currentTf)

    motionProxy.transformInterpolations(effector, frame, path, axisMask, times)

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
