# -*- encoding: UTF-8 -*-
from naoqi import ALProxy

import math
import almath

# Set here your robto's ip.
ip = "127.0.0.1"
# Set here the size of the landmark in meters.
landmarkTheoreticalSize = 0.06 #in meters
# Set here the current camera ("CameraTop" or "CameraBottom").
currentCamera = "CameraTop"

memoryProxy = ALProxy("ALMemory", ip, 9559)
landmarkProxy = ALProxy("ALLandMarkDetection", ip, 9559)

# Subscribe to LandmarkDetected event from ALLandMarkDetection proxy.
landmarkProxy.subscribe("landmarkTest")

# Wait for a mark to be detected.
markData = memoryProxy.getData("LandmarkDetected")
while (len(markData) == 0):
    markData = memoryProxy.getData("LandmarkDetected")

# Retrieve landmark center position in radians.
wzCamera = markData[1][0][0][1]
wyCamera = markData[1][0][0][2]

# Retrieve landmark angular size in radians.
angularSize = markData[1][0][0][3]

# Compute distance to landmark.
distanceFromCameraToLandmark = landmarkTheoreticalSize / ( 2 * math.tan( angularSize / 2))

motionProxy = ALProxy("ALMotion", ip, 9559)

# Get current camera position in NAO space.
transform = motionProxy.getTransform(currentCamera, 2, True)
transformList = almath.vectorFloat(transform)
robotToCamera = almath.Transform(transformList)

# Compute the rotation to point towards the landmark.
cameraToLandmarkRotationTransform = almath.Transform_from3DRotation(0, wyCamera, wzCamera)

# Compute the translation to reach the landmark.
cameraToLandmarkTranslationTransform = almath.Transform(distanceFromCameraToLandmark, 0, 0)

# Combine all transformations to get the landmark position in NAO space.
robotToLandmark = robotToCamera * cameraToLandmarkRotationTransform *cameraToLandmarkTranslationTransform

print "x " + str(robotToLandmark.r1_c4) + " (in meters)"
print "y " + str(robotToLandmark.r2_c4) + " (in meters)"
print "z " + str(robotToLandmark.r3_c4) + " (in meters)"

landmarkProxy.unsubscribe("landmarkTest")

