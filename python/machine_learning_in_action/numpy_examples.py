'''
------------------------------------------------------
Book: Machine Learning In Action
# Lesson: Numpy Library
# Author: Kelly Chan
# Date: Jan 20 2014
------------------------------------------------------
'''


from numpy import *


print "random:"
print random.rand(3,4)
print "\n"

randMatrix = mat(random.rand(3,4))
print "randMatrix:"
print randMatrix
print "\n"

inverRandMatrix = randMatrix.I
print "inverseRandMatrix:"
print inverRandMatrix
print "\n"

identityMatrix = randMatrix * inverRandMatrix
print "IdentityMatrix = randMatrix * inverseRandMatrix:"
print identityMatrix
print "\n"

print "I:"
print eye(3)
print "\n"

print "identityMatrix - I:"
print identityMatrix - eye(3)
print "\n"


print "rows, cols:"
print randMatrix.shape
print "\n"

print "# of rows:"
print randMatrix.shape[0]
print "\n"

print "# of cols:"
print randMatrix.shape[1]
print "\n"
