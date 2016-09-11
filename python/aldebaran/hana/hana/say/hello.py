from naoqi import ALProxy

class Say(object):

    def __init__(self, port, host):
        self.port = port
        self.host = host

    def say_something(self, text):
        tts = ALProxy("ALTextToSpeech", self.port, self.host)
        tts.say(text)
