# -*- encoding: UTF-8 -*-

"""
Receive and display remote buttons event:
"""
import naoqi
import time

# create python module
class myModule(naoqi.ALModule):

    def pythondatachanged(self, strVarName, value, strMessage):
        """callback when data change"""
        print "Data changed on", strVarName, ": ", value, " ", strMessage

# call method
try:
    lircProxy = naoqi.ALProxy("ALInfrared")
    lircProxy.initReception(repeatThreshold = 10)
    pythonModule = myModule("pythonModule")
    memProxy = naoqi.ALProxy("ALMemory")
    memProxy.subscribeToEvent("InfraRedRemoteKeyReceived", "pythonModule",
             "pythondatachanged")
except Exception, e:
    print "error"
    print e
    exit(1)
time.sleep(10)
exit(0)
