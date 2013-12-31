from datetime import datetime

now = datetime.now()
print now
print now.year
print now.month
print now.day
print str(now.month) + "/" + str(now.day) + "/" + str(now.year)
print str(now.hour) + ":" + str(now.minute) + ":" + str(now.second)
