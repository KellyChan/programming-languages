from settings.base import *

from say.hello import Say
from say.reading import Reading

from record.recording import Record

if __name__ == '__main__':


    #hello = Say(HOST, PORT)
    #hello.say_something("Hello, my name is Nao, nice to meet you.")

    #reading = Reading(HOST, PORT)
    #reading.read('./say/texts/coffee_en.txt')

    record = Record(HOST, PORT)
    record.write_data()
