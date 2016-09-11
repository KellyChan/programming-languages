# -*- encoding: UTF-8 -*-
# Before running this command please check your PYTHONPATH is set correctly to the folder of your pynaoqi sdk.
from naoqi import ALProxy 

# Set the IP address of your NAO.
ip = "10.0.252.97"

# Connect to ALSonar module.
sonarProxy = ALProxy("ALSonar", ip, 9559)

# Subscribe to sonars, this will launch sonars (at hardware level) and start data acquisition.
sonarProxy.subscribe("myApplication")

#Now you can retrieve sonar data from ALMemory.
memoryProxy = ALProxy("ALMemory", ip, 9559)

# Get sonar left first echo (distance in meters to the first obstacle).
memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value")

# Same thing for right.
memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value")

# Please read Sonar ALMemory keys section if you want to know the other values you can get.

