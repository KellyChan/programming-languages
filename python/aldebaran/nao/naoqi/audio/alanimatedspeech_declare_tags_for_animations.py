#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

'''Declare tags to animations'''

import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):

    animatedSpeechProxy = ALProxy("ALAnimatedSpeech", robotIP, PORT)

    # associate animation tag "myhellotag" to the animation "animations/Stand/Gestures/Hey_1"
    # associate animation tag "myhellotag" to the animation "myanimlib/Sit/HeyAnim"
    # associate animation tag "cool" to the animation "myanimlib/Sit/CoolAnim"
    tfa = { "myhellotag" : ["animations/Stand/Gestures/Hey_1", "myanimlib/Sit/HeyAnim"],
            "cool" : ["myanimlib/Sit/CoolAnim"] }

    animatedSpeechProxy.declareTagForAnimations(tfa)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
