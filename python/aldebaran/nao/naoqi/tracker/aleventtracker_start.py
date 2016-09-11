# -*- encoding: UTF-8 -*-

"""
This example shows how to use ALTracker to track an object with trackEvent api.
"""

import time
import argparse
from naoqi import ALProxy

def main(IP, PORT):

    print "Connecting to", IP, "with port", PORT
    motion = ALProxy("ALMotion", IP, PORT)
    posture = ALProxy("ALRobotPosture", IP, PORT)
    tracker = ALProxy("ALTracker", IP, PORT)

    # First, wake up.
    motion.wakeUp()

    fractionMaxSpeed = 0.8

    # Go to posture stand
    posture.goToPosture("StandInit", fractionMaxSpeed)

    # Set target to track.
    eventName = "ALTracker/BlobDetected"

    # set mode
    mode = "Move"
    tracker.setMode(mode)

    # Set the robot relative position to target
    # The robot stays a 50 centimeters of target with 10 cm precision
    tracker.setRelativePosition([-0.5, 0.0, 0.0, 0.1, 0.1, 0.3])

    # Then, start tracker.
    tracker.trackEvent(eventName)

    print "ALTracker successfully started."
    print "Use Ctrl+c to stop this script."

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print
        print "Interrupted by user"
        print "Stopping..."

    # Stop tracker, go to posture Sit.
    tracker.stopTracker()
    tracker.unregisterAllTargets()
    posture.goToPosture("Sit", fractionMaxSpeed)
    motion.rest()

    print "ALTracker stopped."


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number.")

    args = parser.parse_args()

    main(args.ip, args.port)
