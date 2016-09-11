# -*- encoding: UTF-8 -*- 

'''Walk To: Small example to make Nao Walk follow'''
'''         a Dubins Curve '''

import sys
import almath as m #python's wrapping of almath
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

    # first we defined the goal
    goal = m.Pose2D(0.0, -0.4, 0.0)

    # We get the dubins solution (control points) by
    # calling an almath function

    circleRadius = 0.04
    # Warning : the circle use by dubins curve
    #           have to be 4*CircleRadius < norm(goal)
    dubinsSolutionAbsolute = m.getDubinsSolutions(goal, circleRadius)

    # moveTo With control Points use relative commands but
    # getDubinsSolution return absolute position
    # So, we compute dubinsSolution in relative way
    dubinsSolutionRelative = []
    dubinsSolutionRelative.append(dubinsSolutionAbsolute[0])
    for i in range(len(dubinsSolutionAbsolute)-1):
        dubinsSolutionRelative.append(
                dubinsSolutionAbsolute[i].inverse() *
                dubinsSolutionAbsolute[i+1])

    # create a vector of moveTo with dubins Control Points
    moveToTargets = []
    for i in range(len(dubinsSolutionRelative)):
        moveToTargets.append(
            [dubinsSolutionRelative[i].x,
             dubinsSolutionRelative[i].y,
             dubinsSolutionRelative[i].theta] )

    # Initialized the Move process and be sure the robot is ready to move
    # without this call, the first getRobotPosition() will not refer to the position
    # of the robot before the move process
    motionProxy.moveInit()

    # get robot position before move
    robotPositionBeforeCommand  = m.Pose2D(motionProxy.getRobotPosition(False))

    motionProxy.moveTo( moveToTargets )

    # With MoveTo control Points, it's also possible to customize the gait parameters
    # motionProxy.moveTo(moveToTargets,
    #                    [["StepHeight", 0.001],
    #                     ["MaxStepX", 0.06],
    #                     ["MaxStepFrequency", 1.0]])

    # get robot position after move
    robotPositionAfterCommand = m.Pose2D(motionProxy.getRobotPosition(False))

    # compute and print the robot motion
    robotMoveCommand = m.pose2DInverse(robotPositionBeforeCommand)*robotPositionAfterCommand
    print "The Robot Move Command: ", robotMoveCommand


if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_moveToDubinsCurve.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)
