from naoqi import ALProxy

if __name__ == '__main__':

    almemory = ALProxy("ALMemory", "nao.local", 9559)
    pings = almemory.ping()

