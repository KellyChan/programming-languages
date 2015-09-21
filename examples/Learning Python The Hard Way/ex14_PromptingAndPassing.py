'''
Created on 2013-8-14
@author: Kelly Chan

Python Version: V3.3

Book: Learn Python The Hard Way
Ex14: Prompting And Passing

'''

import sys
import os

#script, user_name = argv

user_name = os.environ.get( "USERNAME" ) # get system username
script = sys.argv[0] # get script filename
prompt = '> ' # initialization

print("Hi %s, I'm the %s script." % (user_name, script))
print("I'd like to ask you a few questions.")

print("Do you like me %s?" % user_name)
likes = input(prompt)

print("Where do you live %s?" % user_name)
lives = input(prompt)

print("What kind of computer do you have?")
computer = input(prompt)

print("""
Alright, so you said %r about liking me.
You live in %r. Not sure where that is.
And you have a %r computer. Nice.
""" % (likes, lives, computer))