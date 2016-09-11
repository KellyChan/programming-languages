# -*- encoding: UTF-8 -*-
# -*- encoding: UTF-8 -*-

''' Reflexes: LArm Collision Avoidance '''

import motion
import time
import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    ''' Example showing the effect of collision detection
        Nao bumps his chest with his left arm with collision detection enabled
        or disabled.
    '''
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    # Get the robot configuration.
    robotConfig = motionProxy.getRobotConfig()
    robotName = ""
    for i in range(len(robotConfig[0])):
        if (robotConfig[0][i] == "Model Type"):
            robotName = robotConfig[1][i]

    ##################
    # Initialization #
    ##################
    pChainName = "LArm"

    # Send robot to Pose Init.
    moveLArm(motionProxy, robotName, "Init")

    # Disable collision detection on LArm chain.
    pEnable = False
    success = motionProxy.setCollisionProtectionEnabled(pChainName, pEnable)
    if (not success):
        print("Failed to disable collision protection")
    time.sleep(1.0)
    # Make NAO's arm move so that it bumps its torso.
    moveLArm(motionProxy, robotName, "Final")
    time.sleep(1.0)

    # Go back to pose init.
    moveLArm(motionProxy, robotName, "Init")
    # Enable collision detection on chainName.
    pEnable = True
    success = motionProxy.setCollisionProtectionEnabled(pChainName, pEnable)
    if (not success):
        print("Failed to enable collision protection")

    # Make NAO's arm move and see that it does not bump on the torso.
    moveLArm(motionProxy, robotName, "Final")
    time.sleep(1.0)

    # Go back to pose init.
    moveLArm(motionProxy, robotName, "Init")


def moveLArm(motionProxy, pRobotName, pPose):
    ''' Function to make NAO bump on his Torso with his left arm '''

    # Define the name of the chain controlled
    pChainName = "LArm"

    # Define the final position.
    if (pPose == "Init"):
        pTargetAngles = [
             80, # LShoulderPitch
             20, # LShoulderRoll
            -80, # LElbowYaw
            -60] # LElbowRoll
    elif (pPose == "Final"):
        pTargetAngles = [
             50, # LShoulderPitch
              6, # LShoulderRoll
              0, # LElbowYaw
           -150] # LElbowRoll
    else:
        print "ERROR: Your pose is unknown"
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
    pTargetAngles = [x * motion.TO_RAD for x in pTargetAngles]

    # Set the fraction of max speed for the arm movement.
    pMaxSpeedFraction = 0.5

    # Set NAO in stiffness On.
    motionProxy.setStiffnesses("LArm", 1.0)

    # Move the arm to the final position.
    motionProxy.angleInterpolationWithSpeed(pChainName, pTargetAngles, pMaxSpeedFraction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
