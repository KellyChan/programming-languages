def is_prime(x):
    count = 0
    if x < 2:
        return False
    elif (x == 2):
        return True
    else:
        for i in range(2,x):
            if (x % i == 0):
                count += 1
        if count == 0:
            return True
        else:
            return False
