# -*- encoding: UTF-8 -*- 
#!/usr/bin/env python

from naoqi import ALProxy

ROBOT_IP = "127.0.0.1"

alnotificationmanager = ALProxy("ALNotificationManager", ROBOT_IP, 9559)
notifications = alnotificationmanager.notifications()

for notification in notifications:
    notifDict = dict(notification)
    print "Notification ID: " + str(notifDict["id"])
    print "\tMessage: " + notifDict["message"]
    print "\tSeverity: " + notifDict["severity"]
    print "\tRemove On Read: " + str(notifDict["removeOnRead"])
    print "-----------\n"
