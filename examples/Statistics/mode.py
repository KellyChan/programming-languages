def mode(data):
    modecnt = 0
    for i in range(len(data)):
        imode = data.count(data[i])
        if imode > modecnt:
            modecnt = imode
            mode = data[i]
    return mode