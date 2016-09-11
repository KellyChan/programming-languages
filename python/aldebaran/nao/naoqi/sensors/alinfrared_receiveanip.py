# -*- encoding: UTF-8 -*-

"""Receive an IP address"""
import naoqi
import time

# create python module
class myModule(naoqi.ALModule):

    def pythondatachanged(self, strVarName, value, strMessage):
        """callback when data change"""
        print "IP address = ", value, " ", strMessage

# call method
try:
    pythonModule = myModule("pythonModule")
    prox = naoqi.ALProxy("ALMemory")
    prox.subscribeToEvent("InfraRedIpAdressReceived", "pythonModule", "pythondatachanged")
except Exception, e:
    print "error"
    print e
    exit(1)
time.sleep(10)
exit(0)
