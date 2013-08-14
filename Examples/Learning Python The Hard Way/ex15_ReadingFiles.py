'''
Created on 2013-8-14
@author: Kelly Chan

Python Version: V3.3

Book: Learn Python The Hard Way
Ex15: Reading Files

'''

import sys

script = sys.argv[0]
filename = 'G:\eclipseWorkspace\Python\examples\learningPythonTheHardWay\source\ex15.txt'

txt = open(filename)
print("Here's your file %r:" % filename)
print(txt.read())

print("Type the filename again:")
file_again = input("> ")
txt_again = open(file_again)
print(txt_again.read())