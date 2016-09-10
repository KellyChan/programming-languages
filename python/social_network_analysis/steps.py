import math
 
def time(n):
    return 3 + 2 * math.ceil(n/5.0)
 
def countdown(x):
    y = 0
    while x > 0:
        x = x - 5
        y = y + 1
    print y
 
print countdown(50)