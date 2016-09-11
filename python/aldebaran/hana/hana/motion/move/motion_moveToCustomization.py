# -*- encoding: UTF-8 -*- 

'''Move To: Small example to make Nao Move To an Objective '''
'''         With customization '''

import sys
from naoqi import ALProxy


def StiffnessOn(proxy):
    # We use the "Body" name to signify the collection of all joints
    pNames = "Body"
    pStiffnessLists = 1.0
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)


def main(robotIP):
    try:
        motionProxy = ALProxy("ALMotion", robotIP, 9559)
    except Exception, e:
        print "Could not create proxy to ALMotion"
        print "Error was: ", e

    # Set NAO in stiffness On
    StiffnessOn(motionProxy)

    x     = 0.2
    y     = 0.0
    theta = 0.0

    # This example show customization for the both foot
    # with all the possible gait parameters
    motionProxy.moveTo(x, y, theta,
        [ ["MaxStepX", 0.02],         # step of 2 cm in front
          ["MaxStepY", 0.16],         # default value
          ["MaxStepTheta", 0.4],      # default value
          ["MaxStepFrequency", 0.0],  # low frequency
          ["StepHeight", 0.01],       # step height of 1 cm
          ["TorsoWx", 0.0],           # default value
          ["TorsoWy", 0.1] ])         # torso bend 0.1 rad in front

    # This example show customization for the both foot
    # with just one gait parameter, in this case, the other
    # parameters are set to the default value
    motionProxy.moveTo(x, y, theta, [ ["StepHeight", 0.04] ]) # step height of 4 cm


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_moveToCustomization.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
