# -*- encoding: UTF-8 -*-
# This test demonstrates how to use the ALVisionRecognition module.
# Note that you might not have this module depending on your distribution.
#
# - We first instantiate a proxy to the ALVisionRecognition module
#     Note that this module should be loaded on the robot's NaoQi.
#     The module output its results in ALMemory in a variable
#     called "PictureRecognized".

# - We then read this AlMemory value and check whether we get
#   interesting things.


import time

from naoqi import ALProxy
from naoqi import ALBroker
from naoqi import ALModule

count = 10
period = 500
moduleName = "pythonModule"
NAO_IP = "nao.local"  # Replace here with your NaoQi's IP address.
PC_IP = "127.0.0.1"   # Replace here with your computer IP address
PORT = 9559
memValue = "PictureDetected" # ALMemory variable where the ALVisionRecognition module outputs its results.


# create python module
class myModule(ALModule):
  """python class myModule test auto documentation"""

  def pictureChanged(self, strVarName, value, strMessage):
    """callback when data change"""
    print "datachanged", strVarName, " ", value, " ", strMessage
    global count
    count = count-1

broker = ALBroker("pythonBroker", PC_IP,9999, NAO_IP,9559)
pythonModule = myModule(moduleName)

# Create a proxy to ALMemory
try:
  memoryProxy = ALProxy("ALMemory", NAO_IP, PORT)
except RuntimeError,e:
  print "Error when creating ALMemory proxy:"
  exit(1)


# Have the python module called back when picture recognition results change.
try:
  memoryProxy.subscribeToEvent(memValue, moduleName, "pictureChanged")
except RuntimeError,e:
  print "Error when subscribing to micro event"
  exit(1)


# Let the picture recognition run for a little while (will stop after 'count' calls of the callback).
# You can check the results using a browser connected on your Nao, then
# Advanced -> Memory -> type PictureDetected in the field
while count>0:
  time.sleep(1)


# unsubscribe modules
memoryProxy.unsubscribeToEvent(memValue, moduleName)
#recoProxy.unsubscribe(moduleName)


print 'end of vision_recognition python script'

