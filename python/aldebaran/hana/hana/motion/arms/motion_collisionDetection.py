# -*- encoding: UTF-8 -*- 

''' Collision detection : arm collision detection '''

import sys
import motion
import time
from naoqi import ALProxy


def StiffnessOn(proxy):
    # We use the "Body" name to signify the collection of all joints
    pNames = "Body"
    pStiffnessLists = 1.0
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)


def moveArm(motionProxy, pTarget, pRobotName, pChainName):
    ''' Function to make NAO bump on his Torso or Head with his arm '''

    # Set the fraction of max speed for the arm movement.
    pMaxSpeedFraction = 0.5

    # Define the final position.
    if (pTarget == "Torso"):
        ShoulderPitchAngle = 50
    elif (pTarget == "Head"):
        ShoulderPitchAngle = -50
    else:
        print "ERROR: target is unknown"
        print "Must be Torso or Head"
        print "---------------------"
        exit(1)

    ShoulderRollAngle  = 6
    ElbowYawAngle      = 0
    ElbowRollAngle     = -150

    if pChainName == "LArm":
        pTargetAngles = [ShoulderPitchAngle, +ShoulderRollAngle,
            +ElbowYawAngle, +ElbowRollAngle]
    elif pChainName == "RArm":
        pTargetAngles = [ShoulderPitchAngle, -ShoulderRollAngle,
            -ElbowYawAngle, -ElbowRollAngle]
    else:
        print "ERROR: chainName is unknown"
        print "Must be LArm or RArm"
        print "---------------------"
        exit(1)

    # Set the target angles according to the robot version.
    if (pRobotName == "naoH25") or\
       (pRobotName == "naoAcademics") or\
       (pRobotName == "naoT14"):
        pTargetAngles += [0.0, 0.0]
    elif (pRobotName == "naoH21"):
        pass
    elif (pRobotName == "naoT2"):
        pTargetAngles = []
    else:
        print "ERROR: Your robot is unknown"
        print "This test is not available for your Robot"
        print "---------------------"
        exit(1)

    # Convert to radians.
    pTargetAngles = [ x * motion.TO_RAD for x in pTargetAngles]

    # Move the arm to the final position.
    motionProxy.angleInterpolationWithSpeed(
        pChainName, pTargetAngles, pMaxSpeedFraction)


def main(robotIP, pChainName):
    ''' Example showing the effect of collision detection
        Nao bumps his chest with his left arm with collision detection enabled
        or disabled.
    '''

    ##################
    # Initialization #
    ##################

   # Init proxies.
    try:
        motionProxy = ALProxy("ALMotion", robotIP, 9559)
    except Exception, e:
        print "Could not create proxy to ALMotion"
        print "Error was: ", e

    try:
        postureProxy = ALProxy("ALRobotPosture", robotIP, 9559)
    except Exception, e:
        print "Could not create proxy to ALRobotPosture"
        print "Error was: ", e

    # Set NAO in Stiffness On
    StiffnessOn(motionProxy)

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    # Get the robot configuration.
    robotConfig = motionProxy.getRobotConfig()
    robotName = ""
    for i in range(len(robotConfig[0])):
        if (robotConfig[0][i] == "Model Type"):
            robotName = robotConfig[1][i]

    ###############################
    # Arm motion bumping on torso #
    ###############################

    # Disable collision detection on chainName.
    pEnable = False
    success = motionProxy.setCollisionProtectionEnabled(pChainName, pEnable)
    if (not success):
        print("Failed to disable collision protection")
    time.sleep(1.0)

    # Make NAO's arm move so that it bumps its torso.
    pTargetName = "Torso"
    moveArm(motionProxy, pTargetName, robotName, pChainName)
    time.sleep(1.0)

    # Go back to pose init.
    postureProxy.goToPosture("StandInit", 1.0)

    # Enable collision detection on chainName.
    pEnable = True
    success = motionProxy.setCollisionProtectionEnabled(pChainName, pEnable)
    if (not success):
        print("Failed to enable collision protection")
    time.sleep(1.0)

    # Make NAO's arm move and see that it does not bump on the torso.
    pTargetName = "Torso"
    moveArm(motionProxy, pTargetName, robotName, pChainName)

    ##############################
    # Arm motion bumping on head #
    ##############################

    time.sleep(1.0)
    # Go back to pose init.
    postureProxy.goToPosture("StandInit", 1.0)
    # Disable collision detection on chainName.
    pEnable = False
    success = motionProxy.setCollisionProtectionEnabled(pChainName, pEnable)
    if (not success):
        print("Failed to disable collision protection")
    time.sleep(1.0)
    # Make NAO's arm move so that it bumps its head.
    pTargetName = "Head"
    moveArm(motionProxy, pTargetName, robotName, pChainName)

    time.sleep(1.0)
    # Go back to pose init.
    postureProxy.goToPosture("StandInit", 1.0)
    # Enable collision detection on chainName.
    pEnable = True
    success = motionProxy.setCollisionProtectionEnabled(pChainName, pEnable)
    if (not success):
        print("Failed to enable collision protection")
    # Make NAO's arm move and see that it does not bump on the head.
    pTargetName = "Head"
    moveArm(motionProxy, pTargetName, robotName, pChainName)

    time.sleep(1.0)
    # Go back to pose init.
    postureProxy.goToPosture("StandInit", 1.0)

if __name__ == "__main__":
    robotIp = "127.0.0.1"
    effectorName = "LArm"

    if (len(sys.argv) <= 1):
      print "Usage python motion_collisionDetection.py robotIP(optional default: 127.0.0.1) effectorName(RArm or LArm)"

    else:
        if (len(sys.argv) > 1):
            robotIp = sys.argv[1]

        if (len(sys.argv) > 2):
            effectorName = sys.argv[2]

    main(robotIp, effectorName)
