'''
Created on 2013-8-14
@author: Kelly Chan

Python Version: V3.3

Book: Learn Python The Hard Way
Ex17: More Files

'''

import sys
from os.path import exists

script = sys.argv[0]
from_file = 'G:\eclipseWorkspace\Python\examples\learningPythonTheHardWay\source\ex15.txt'
to_file = 'G:\eclipseWorkspace\Python\examples\learningPythonTheHardWay\source\ex17.txt'


print("Copying from %s to %s" % (from_file, to_file))
# we could do these two on one line too, how?
in_file = open(from_file)
indata = in_file.read()

print("The input file is %d bytes long" % len(indata))
print("Does the output file exist? %r" % exists(to_file))
print("Ready, hit RETURN to continue, CTRL-C to abort.")
input()
out_file = open(to_file, 'w')
out_file.write(indata)
print("Alright, all done.")
out_file.close()

in_file.close()