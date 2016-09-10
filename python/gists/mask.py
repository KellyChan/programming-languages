def check_bit4(n):
    mask = 0b1000
    desired = n & mask
    if desired:
        return "on"
    else:
        return "off"
