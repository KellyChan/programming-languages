# -*- encoding: UTF-8 -*-
"""
This is a simple python script to easily record .arv files.

We create a video client to get images.
We call "recordVideo" on our video client, so that each image that is retrieved
by the client is saved to disk.

"""


import time

from naoqi import ALProxy


IP = "nao.local"  # Enter your NAOqi's IP address here.
PORT=9559

#________________________________
# Generic Proxy creation
#________________________________

camProxy = ALProxy("ALVideoDevice",IP,PORT)

#________________________________
# Vision module creation
#________________________________

resolution = 1 #0:QVGA 1:QVGA 2:VGA (camera doesn't provide QQVGA that is obtained through CPU processing)
colorSpace = 9 #kYUV422InterlacedColorSpace
fps = 30       #not activated

# We subscrive our video client.
nameId = "pythonGVM"
nameId = camProxy.subscribe(nameId, resolution, colorSpace, fps)

#________________________________
# Ask to grab a maximum of 3500
# images obtained by the vision module
# After this call, all images we will retrieved using "nameId" will be saved
# to disk.
#________________________________

recording = camProxy.recordVideo(nameId, "/home/nao/naoqi/001_VGA", 3500, 1)

print  "launch recording"
print recording

#________________________________
# The vision module requests 300 images
#________________________________

for i in range(0, 500):
  image = camProxy.getImageLocal(nameId)
  camProxy.releaseImage(nameId)
  time.sleep(0.003)

#________________________________
# Stop manually the recording
# after 300 images
#________________________________

print "finishing"
camProxy.stopVideo(nameId)
camProxy.unsubscribe(nameId)

print 'end of script'
