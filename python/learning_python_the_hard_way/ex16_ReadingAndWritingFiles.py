'''
Created on 2013-8-14
@author: Kelly Chan

Python Version: V3.3

Book: Learn Python The Hard Way
Ex16: Reading And Writing Files

'''

import sys

script = sys.argv[0]
filename = 'G:\eclipseWorkspace\Python\examples\learningPythonTheHardWay\source\ex15.txt'

print("We're going to erase %r." % filename)
print("If you don't want that, hit CTRL-C (^C).")
print("If you do want that, hit RETURN.")
input("?")

print("Opening the file...")
target = open(filename, 'w')
print("Truncating the file. Goodbye!")
target.truncate()

print("Now I'm going to ask you for three lines.")
line1 = input("line 1: ")
line2 = input("line 2: ")
line3 = input("line 3: ")

print("I'm going to write these to the file.")
target.write(line1)
target.write("\n")
target.write(line2)
target.write("\n")
target.write(line3)
print("And finally, we close it.")
target.close()