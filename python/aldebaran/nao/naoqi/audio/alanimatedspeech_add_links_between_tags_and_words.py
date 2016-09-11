#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

'''Add some links between tags and words'''

import argparse
from naoqi import ALProxy

def main(robotIP, PORT=9559):

    animatedSpeechProxy = ALProxy("ALAnimatedSpeech", robotIP, PORT)

    # associate word "hey" with animation tag "hello"
    # associate word "yo" with animation tag "hello"
    # assiciate word "everybody" with animation tag "everything"
    ttw = { "hello" : ["hey", "yo"],
            "everything" : ["everybody"] }

    animatedSpeechProxy.addTagsToWords(ttw)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)
