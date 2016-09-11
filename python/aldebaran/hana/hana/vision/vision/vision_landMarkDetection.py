# -*- encoding: UTF-8 -*-
# This test demonstrates how to use the ALLandMarkDetection module.
# Note that you might not have this module depending on your distribution
#
#  /!\  Please change the IP address with your robot IP.
#
# - We first instantiate a proxy to the ALLandMarkDetection module
#     Note that this module should be loaded on the robot's naoqi.
#     The module output its results in ALMemory in a variable
#     called "LandmarkDetected"

# - We then read this ALMemory value and check whether we get
#   interesting things.

import time

from naoqi import ALProxy

IP = "nao.local"  # Replace here with your NaoQi's IP address.
PORT = 9559

# Create a proxy to ALLandMarkDetection
try:
  landMarkProxy = ALProxy("ALLandMarkDetection", IP, PORT)
except Exception, e:
  print "Error when creating landmark detection proxy:"
  print str(e)
  exit(1)

# Subscribe to the ALLandMarkDetection proxy
# This means that the module will write in ALMemory with
# the given period below
period = 500
landMarkProxy.subscribe("Test_LandMark", period, 0.0 )

# ALMemory variable where the ALLandMarkdetection modules
# outputs its results
memValue = "LandmarkDetected"

# Create a proxy to ALMemory
try:
  memoryProxy = ALProxy("ALMemory", IP, PORT)
except Exception, e:
  print "Error when creating memory proxy:"
  print str(e)
  exit(1)


# A simple loop that reads the memValue and checks whether landmarks are detected.
for i in range(0, 20):
  time.sleep(0.5)
  val = memoryProxy.getData(memValue)

  print ""
  print "*****"
  print ""

  # Check whether we got a valid output.
  if(val and isinstance(val, list) and len(val) >= 2):

    # We detected naomarks !
    # For each mark, we can read its shape info and ID.

    # First Field = TimeStamp.
    timeStamp = val[0]

    # Second Field = array of Mark_Info's.
    markInfoArray = val[1]

    try:
      # Browse the markInfoArray to get info on each detected mark.
      for markInfo in markInfoArray:

        # First Field = Shape info.
        markShapeInfo = markInfo[0]

        # Second Field = Extra info (ie, mark ID).
        markExtraInfo = markInfo[1]
        print "mark  ID: %d" % (markExtraInfo[0])
        print "  alpha %.3f - beta %.3f" % (markShapeInfo[1], markShapeInfo[2])
        print "  width %.3f - height %.3f" % (markShapeInfo[3], markShapeInfo[4])

    except Exception, e:
      print "Naomarks detected, but it seems getData is invalid. ALValue ="
      print val
      print "Error msg %s" % (str(e))
  else:
      print "No landmark detected"


# Unsubscribe the module.
landMarkProxy.unsubscribe("Test_LandMark")

print "Test terminated successfully."
