# -*- encoding: UTF-8 -*-

import argparse
import motion
import almath
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)

    # Wake up robot
    motionProxy.wakeUp()

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    # Get transform of Left Arm in Torso frame
    chainName = "LArm"
    frame     = motion.FRAME_TORSO
    useSensor = False
    tf = almath.Transform(motionProxy.getTransform(chainName, frame, useSensor))

    # Compute desired transform: rotation of -20 degrees around the Z axis
    tfEnd = almath.Transform.fromRotZ(-20.0*almath.TO_RAD)*tf
    tfEnd.r1_c4 = tf.r1_c4
    tfEnd.r2_c4 = tf.r2_c4
    tfEnd.r3_c4 = tf.r3_c4

    # Set the desired target
    axisMask = 63 # rotation
    fractionMaxSpeed = 0.1
    transform = [val for val in tfEnd.toVector()]
    motionProxy.setTransforms(chainName, frame, transform, fractionMaxSpeed, axisMask)

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
