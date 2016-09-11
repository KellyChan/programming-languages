# -*- encoding: UTF-8 -*-
""" 
Record some sensors values and write them into a file.
"""

import os
import sys
import time

from naoqi import ALProxy

# MEMORY_VALUE_NAMES is the list of ALMemory values names you want to save.
ALMEMORY_KEY_NAMES = [
"Device/SubDeviceList/HeadYaw/Position/Sensor/Value",
"Device/SubDeviceList/HeadYaw/Position/Actuator/Value",
]


class Record(object):

    def __init__(self, host, port):
        self.host = host
        self.port = port

    def record(self):
        """
        Returns a matrix of values of recording the data from ALMemory.
        """
        
        print "Recording data ..."
        memory = ALProxy("ALMemory", self.host, self.port)

        data = list()
        for i in range(1, 100):
            line = list()
            for key in ALMEMORY_KEY_NAMES:
                value = memory.getData(key)
                line.append(value)
            data.append(line)
            time.sleep(0.05)
        return data


    def write_data(self):
        motion = ALProxy("ALMotion", self.host, self.port)
        # set stiffness on for Head motors
        motion.setStiffnesses("Head", 1.0)
        # will go to 1.0 then 0 radian
        # in two seconds
        motion.post.angleInterpolation(["HeadYaw"],
                                       [1.0, 0.0],
                                       [1, 2],
                                       False)

        data = self.record()
        motion.setStiffnesses("Head", 0.0)
        output = os.path.abspath("record.csv")
        
        with open(output, "w") as fp:
            for line in data:
                fp.write("; ".join(str(x) for x in line))
                fp.write("\n")
  
        print "Results written to", output




