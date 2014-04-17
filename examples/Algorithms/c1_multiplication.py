# z = ab
# ab = xy + z
def naive(a, b):
    x = a; y = b
    z = 0
    while x > 0:
        z = z + y
        x = x - 1
    return z

print naive(4, 5)


def russian(a, b):
    x = a; y = b
    z = 0
    while x > 0:
        if x % 2 == 1: z = z + y
        y = y << 1  # y * 2 | <<: shift
        x = x >> 1  # x / 2 | 17/10001 --> 8/1000
    return z


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


def time(a):
    # The number of steps it takes to execute naive(a, b)
    # as a function of a
    steps = 0
    steps = 3 + 2.0 * a
    return steps
