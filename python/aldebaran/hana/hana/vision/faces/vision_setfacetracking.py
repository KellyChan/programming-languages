# -*- encoding: UTF-8 -*-
# This python script assumes that you have correctly set your PYTHONPATH
# environment variable to "your_naoqi_sdk_path"/lib/.
#
# When tracking is activated, faces looking sideways, or located further away
# should be tracked for a longer period.
# Launch Monitor, Cameraviewer, activate face detection, and see if it works better.
#

from naoqi import ALProxy

import sys


USAGE = "USAGE:\n" \
        "python vision_setfacetracking.py [NAO_IP] [0 or 1] \n" \
        "\nExamples: \n" \
        "Enable tracking: set_tracking.py 192.168.1.102 1\n" \
        "Disable tracking: set_tracking.py 192.168.1.102 0"


def set_nao_face_detection_tracking(nao_ip, nao_port, tracking_enabled):
    """Make a proxy to nao's ALFaceDetection and enable/disable tracking.

    """
    faceProxy = ALProxy("ALFaceDetection", nao_ip, nao_port)

    print "Will set tracking to '%s' on the robot ..." % tracking_enabled

    # Enable or disable tracking.
    faceProxy.enableTracking(tracking_enabled)

    # Just to make sure correct option is set.
    print "Is tracking now enabled on the robot?", faceProxy.isTrackingEnabled()


def main():
    # Specify your IP address here.
    nao_ip = "127.0.0.1"
    nao_port = 9559

    tracking_enabled = True

    try:
        if len(sys.argv) > 1:
            nao_ip = sys.argv[1]

        if len(sys.argv) > 2:
            tracking_enabled = bool(int(sys.argv[2]))

        set_nao_face_detection_tracking(nao_ip, nao_port, tracking_enabled)

    except Exception as e:
        print "Exception Caught: %s\n" % e
        print USAGE


if __name__ == "__main__":
    main()


