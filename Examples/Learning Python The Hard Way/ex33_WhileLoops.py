'''
Created on 2013-8-14
@author: Kelly Chan

Python Version: V3.3

Book: Learn Python The Hard Way
Ex33: While Loops

'''

i = 0
numbers = []

while i < 6:
    print("At the top i is %d" % i)
    numbers.append(i)
    
    i = i + 1
    print("Numbers now: ", numbers)
    print("At the bottom i is %d" % i)

print("The numbers: ")
for num in numbers:
    print(num)