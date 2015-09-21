# multiply - ab = xy + z
# total = not_done + done (sum) <- divide and conquer
# divide and conquer -> recursion
 
# naive
def naive(a,b):
    x = a, y = b,
    z = 0
    while x > 0:
        z = z + y
        x = x - 1
    return z
    
# recursion
def rec_naive(a,b):
    if a == 0:
        return 0
    return b + rec_naive(a-1, b)