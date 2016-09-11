# -*- encoding: UTF-8 -*-

import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    # Example showing how to get the mass of "HeadYaw".
    pName = "HeadYaw"
    mass = motionProxy.getMass(pName)
    print pName + " mass: " + str(mass)

    # Example showing how to get the mass "LLeg" chain.
    pName = "LLeg"
    print "LLeg mass: ", motionProxy.getMass(pName)

    # It is equivalent to the following script
    pNameList = motionProxy.getBodyNames("LLeg")
    mass = 0.0
    for pName in pNameList:
        jointMass = motionProxy.getMass(pName)
        print pName + " mass: " + str(jointMass)
        mass = mass + jointMass
    print "LLeg mass:", mass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
