# -*- encoding: UTF-8 -*-

# This is just an example script that shows how images can be accessed
# through ALVideoDevice in python.
# Nothing interesting is done with the images in this example.

from naoqi import ALProxy
import vision_definitions

IP = "nao.local"  # Replace here with your NAOqi's IP address.
PORT = 9559

####
# Create proxy on ALVideoDevice

print "Creating ALVideoDevice proxy to ", IP

camProxy = ALProxy("ALVideoDevice", IP, PORT)

####
# Register a Generic Video Module

resolution = vision_definitions.kQVGA
colorSpace = vision_definitions.kYUVColorSpace
fps = 30

nameId = camProxy.subscribe("python_GVM", resolution, colorSpace, fps)
print nameId

print 'getting images in local'
for i in range(0, 20):
  camProxy.getImageLocal(nameId)
  camProxy.releaseImage(nameId)

resolution = vision_definitions.kQQVGA
camProxy.setResolution(nameId, resolution)

print 'getting images in remote'
for i in range(0, 20):
  camProxy.getImageRemote(nameId)

camProxy.unsubscribe(nameId)

print 'end of gvm_getImageLocal python script'
