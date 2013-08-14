'''
Created on 2013-8-14
@author: Kelly Chan

Python Version: V3.3

Book: Learn Python The Hard Way
Ex6: Strings And Text

'''


binary = "binary"
do_not = "don't"
x = "There are %d types of people." % 10
y = "Those who know %s and those who %s." % (binary, do_not)
print(x)
print(y)
print("I said: %r." % x)
print("I also said: '%s'." % y)

hilarious = False
joke_evaluation = "Isn't that joke so funny?! %r"
print(joke_evaluation % hilarious)

w = "This is the left side of..."
e = "a string with a right side."
print(w + e)