# -*- coding:utf-8 -*-
__author__  = 'KellyChan'
__date__    = '2014-09-07'
__version__ = '1.0.0'


def encode():

    # Morse Table
    CODE = {
            # 26 letters
            'A': '.-',     'B': '-...',   'C': '-.-.',
            'D': '-..',    'E': '.',      'F': '..-.',
            'G': '--.',    'H': '....',   'I': '..',
            'J': '.---',   'K': '-.-',    'L': '.-..',
            'M': '--',     'N': '-.',     'O': '---',
            'P': '.--.',   'Q': '--.-',   'R': '.-.',
            'S': '...',    'T': '-',      'U': '..-',
            'V': '...-',   'W': '.--',    'X': '-..-',
            'Y': '-.--',   'Z': '--..',
     
            # 10 digits
            '0': '-----',  '1': '.----',  '2': '..---',
            '3': '...--',  '4': '....-',  '5': '.....',
            '6': '-....',  '7': '--...',  '8': '---..',
            '9': '----.',
     
            # 16 punctuations
            ',': '--..--', '.': '.-.-.-', ':': '---...', ';': '-.-.-.',
            '?': '..--..', '=': '-...-',  "'": '.----.', '/': '-..-.',
            '!': '-.-.--', '-': '-....-', '_': '..--.-', '(': '-.--.',
            ')': '-.--.-', '$': '...-..-','&': '. . . .','@': '.--.-.'
     
            # you can define more
     
    }

    return CODE

def decode(CODE):
    return dict(map(lambda t:(t[1], t[0]), CODE.items()))


def convertString2Morse(msg):
    morse = ''

    print "Your message was: %s" % msg
    for char in msg:
        if char == ' ':
            morse += ' '
        else:
            # upper(): convert all lowercases to uppercases
            CODE = encode()
            morse += CODE[char.upper()] + ' '
    print "Your Morse Code is: %s" % morse

    return morse

def convertMorse2String(morse):

    string = ''

    codes = morse.split(' ')
    print "Your Morse Code was: %s" % codes

    for code in codes:
        if code == '':
            string += ' '
        else:
            UNCODE = decode(encode())
            string += UNCODE[code]
    print "Your Message is: %s" % string

    return string


def main():

    #msg = raw_input('Please input your message: ')
    msg = "Happy Mid-Autumn Day!"

    morse = convertString2Morse(msg)
    string = convertMorse2String(morse)


if __name__ == '__main__':
    main()

 
 
