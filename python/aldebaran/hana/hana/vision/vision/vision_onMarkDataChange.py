# -*- encoding: UTF-8 -*-
##############################################################################
# Name: vision_onMarkDataChange.py
#
# Summary: Test the ALLandMarkDetection module using a callback mechanism.
#
# Usage: vision_onMarkDataChange.py [Naoqi IP address, local IP address]
#        - Naoqi IP: address Naoqi is listening to.
#        - local IP: address where you launch the script and that Naoqi can
#          use to call you back.
#
# Description:
#   - Create a Python broker using the provided Naoqi and local addresses.
#   - Start the LandMarkDetection extractor by subscribing to it.
#   - Create an ALModule object (markHandler) with its call back function.
#   - Call ALMemory's subscribeOnDataChange so that markHandler.onMarkChange()
#     is called whenever the detection results change.
#   - Wait for some time. Msg is printed if we detect naomarks.
#   - Unsubscribe from LandMarkDetection.
#
##############################################################################

# Used in debug logs.
testName = "python: vision_onMarkDataChange: "

import sys
import time

# IP and PORT default values.
LOCAL_IP = "127.0.0.1" # Replace here with your computer IP address.
LOCAL_PORT = 9999
IP = "nao.local"  # Replace here with your NaoQi's IP address.
PORT = 9559

# Read IP address from first argument if any.
if len( sys.argv ) > 1:
  IP = sys.argv[1]

if len( sys.argv ) > 2:
  LOCAL_IP = sys.argv[2]

########################################################
from naoqi import ALProxy, ALModule, ALBroker


##############################################################################
# Definition of our python module.
class MarkHandlerModule(ALModule):
  """ The main point here is to declare a module with a call back function
      that is called by ALMemory whenever the landmark's results change. """

  # Call back function registered with subscribeOnDataChange that handles
  # changes in LandMarkDetection results.
  def onMarkChange(self, dataName, value, msg):
    """ Call back method called when naomark detection updates its results. """
    print str(value)
    if (len(value) != 0):
      print "We detected naomarks !"

##############################################################################


testFailed = 0

# Create a python broker on the local machine.
broker = ALBroker("pythonBroker", LOCAL_IP, LOCAL_PORT, IP, PORT)

# Creating a proxy to ALLandMarkDetection
try:
  landMarkProxy = ALProxy("ALLandMarkDetection", IP, PORT)
except Exception,e:
  print "Error when creating ALLandMarkDetection proxy:"
  print str(e)
  exit(1)


# Subscribe to the ALLandMarkDetection module.
# This means that the module will write its results in memValue with
# the given period below.
subscriptionPeriod = 500

print "%s : Subscribe to the ALLandMarkDetection proxy..." % (testName)
try:
  landMarkProxy.subscribe("Test_LandMark",  subscriptionPeriod , 1.0 )
  print "%s : Subscribe to the ALLandMarkDetection proxy... OK" % (testName)
except Exception, e:
  print "%s Error :"  % (testName)
  print str(e)
  testFailed = 1


# ALMemory variable where the ALLandMarkdetection module outputs its results.
memValue = "LandmarkDetected"

try:
  markHandlerName = "markHandler"
  # Create our module object.
  markHandler = MarkHandlerModule(markHandlerName)

  # Have module.onMarkChange called back when detection results change.
  memoryProxy = ALProxy("ALMemory")
  memoryProxy.subscribeToEvent(memValue, markHandlerName, "onMarkChange")

  # Let the NaoMark detection run for a little while. You can check the results using
  # a browser connected on your Nao, then Advanced -> Memory -> type mark in the field
  time.sleep(20)

  memoryProxy.unsubscribeToEvent(memValue, markHandlerName)

  # Shut the Python Broker down
  broker.shutdown()

except Exception, e:
  print "%s Error:"  %(testName)
  print str(e)
  testFailed = 1

if (testFailed == 1):
  print "%s : Failed" % (testName)
  exit(1)

print "%s : Success" % (testName)
