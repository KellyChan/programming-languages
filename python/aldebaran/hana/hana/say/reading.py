import codecs

from naoqi import ALProxy


class Reading(object):

    def __init__(self, port, host):
        self.port = port
        self.host = host

    def read(self, text):
        tts = ALProxy("ALTextToSpeech", self.port, self.host)        
        say_from_file(tts, text, 'utf-8')


def say_from_file(tts, filename, encoding):

    with codecs.open(filename, encoding=encoding) as fp:
        contents = fp.read()
        # warning: print contents won't work
        to_say = contents.encode("utf-8")
    tts.say(to_say)
