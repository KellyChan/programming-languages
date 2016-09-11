# -*- encoding: UTF-8 -*-

"""
Send an element of each of the 3 supported data type to nao
"""
from naoqi import ALProxy
import time

lirc=ALProxy("ALInfrared","127.0.0.1",9559) # change IP with NAO IP
lirc.sendIpAddress("127.0.0.1") # Send IP address
time.sleep(0.5) # Delay necessary for a reliable transfer
lirc.send8(42) # Send the number 42
time.sleep(0.5)
lirc.send32(0x42, 0x2A, 0x13, 0x0D) # Send one 32 bits number
time.sleep(0.5)
lirc.send32("36757575") # Send one 32 bits number
