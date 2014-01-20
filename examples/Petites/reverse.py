def reverse(text):
    l = []
    r = ""
    for i in range(len(text)-1, -1, -1):
        l.append(text[i])
    
    for i in range(len(l)):
        r += str(l[i])
    return r
