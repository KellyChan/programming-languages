from naoqi import ALProxy
import time

ROBOT_IP="your.robot.ip.here"

barcode=ALProxy("ALBarcodeReader", ROBOT_IP, 9559)
barcode.subscribe("test_barcode")

memory=ALProxy("ALMemory", ROBOT_IP, 9559)
# Query last data from ALMemory twenty times

for i in range(20):
  data = memory.getData("BarcodeReader/BarcodeDetected")
  print data
  time.sleep(1)
