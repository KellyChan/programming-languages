from naoqi import ALProxy
import time

ip_robot = "127.0.0.1"
port_robot = 9559

basic_awareness = ALProxy("ALBasicAwareness", ip_robot, port_robot)
motion = ALProxy("ALMotion", ip_robot, port_robot)

#wake up
motion.wakeUp()

#start basic_awareness
basic_awareness.startAwareness()

#some time to play with the robot
time.sleep(30)

#stop basic_awareness
basic_awareness.stopAwareness()

#rest
motion.rest()
