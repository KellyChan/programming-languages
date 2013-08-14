'''
Created on 2013-8-14
@author: Kelly Chan

Python Version: V3.3

Book: Learn Python The Hard Way
Ex13: Parameters, Unpacking, Variables

'''


import sys

print("The script name is called:", sys.argv[0])

if len(sys.argv) > 1:
    print("there are", len(sys.argv)-1, "arguments:")
    for arg in sys.argv[1:]:
        print(arg)
else:
    print("there are no arguments!")
    
   