"""
Project: Generating all 4-digit passwords combining digits 0-9, uppercases A-Z and
lowercases a-z

Author: Kelly Chan
Date: Sept 7 2014
"""

import os


def main():

    chars=[
                    '0','1','2','3','4','5','6','7','8','9',
                    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                ]

    base = len(chars)  # 62
    end = len(chars)**4

    outputs = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

    with open(outputs+"/dict.txt", 'w') as f:
        
        for i in range(0, end):
            n = i
            char0 = chars[n%base]
            n = n / base
            char1 = chars[n%base]
            n = n / base
            char2 = chars[n%base]
            n = n / base
            char3 = chars[n%base]
            #print i, char3, char2, char1, char0
            f.write(char3+char2+char1+char0+'\n')

    f.close()

    with open(outputs+"/dict.txt", 'rb') as f:
        for line in f.readlines()[-100:]:
            print line


if __name__ == '__main__':
    main()


