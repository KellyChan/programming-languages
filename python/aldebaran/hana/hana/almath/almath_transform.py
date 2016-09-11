# -*- encoding: UTF-8 -*- 

''' Example showing how to use almath with python and send the results to
    the robot by using a proxy to ALMotion '''

import sys
import time

from naoqi import ALProxy

import almath


def main(IP):
    PORT = 9559

    # Create a proxy to ALMotion.
    try:
        motionProxy = ALProxy("ALMotion", IP, PORT)
    except Exception,e:
        print "Could not create proxy to ALMotion"
        print "Error was: ",e

    chainName = "RArm"
    space = 1
    useSensors = True

    # Set stiffness on.
    motionProxy.stiffnessInterpolation("Body", 1.0, 0.5)

    # Stand up.
    motionProxy.moveInit()

    ##############################################
    # Retrieve a transform matrix using ALMotion #
    ##############################################

    # Retrieve current transform from ALMotion.
    # Convert it to a transform matrix for ALMath.
    origTransform = almath.Transform(
        motionProxy.getTransform(chainName, space, useSensors))

    # Visualize the transform using overriden print from ALMath
    print "Original transform"
    print origTransform

    ##########################################################
    # Use almath to do some computations on transform matrix #
    ##########################################################

    # Compute a transform corresponding to the desired move
    # (here, move the chain for 5cm along the Z axis and the X axis).
    moveTransform = almath.Transform.fromPosition(0.05, 0.0, 0.05)

    # Compute the corresponding target transform.
    targetTransform = moveTransform * origTransform

    # Visualize it.
    print "Target transform"
    print targetTransform

    ##############################################
    # Send a transform to the robot via ALMotion #
    ##############################################

    # Convert it to a tuple.
    targetTransformList = list(targetTransform.toVector())

    # Send the target transform to NAO through ALMotion.
    fractionOfMaxSpeed = 0.5
    axisMask = almath.AXIS_MASK_VEL # Translation X, Y, Z
    motionProxy.setTransform(
        chainName,
        space,
        targetTransformList,
        fractionOfMaxSpeed,
        axisMask)

    time.sleep(2.0)


if __name__ == "__main__":
    if (len(sys.argv)<2):
        print "Usage python almath_transform.py robotIP"
        sys.exit(1)

    main(sys.argv[1])
