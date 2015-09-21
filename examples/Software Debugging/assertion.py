import math
import random

def square_root(x, eps=10e-7):
    assert x >= 0
    y = math.sqrt(x)
    assert abs(y * y - x) < eps # set up a suitable postcondition
    return y

for i in range(1, 1000):
    r = random.random() * 1000
    z = square_root(r)

print "Done!"
