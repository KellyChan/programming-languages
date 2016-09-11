from naoqi import ALProxy

if __name__ == '__main__':

    motion = ALProxy("ALMotion", "nao.local", 9559)
    tts = ALProxy("ALTextToSpeech", "nao.local", 9559)

    motion.setStiffnesses("Body", 1.0)
    
    motion.moveInit()
    motion.moveTo(0.5, 0, 0)

    id = motion.post.moveTo(0.5, 0, 0)
    motion.wait(id, 0)
    tts.say("I am walking")
