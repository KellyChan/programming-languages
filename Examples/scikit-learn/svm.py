from sklearn import datasets

print "loading iris data:"
iris = datasets.load_iris()
print "iris: (rows, cols)"
print iris.data.shape
print "iris: (rows,)"
print iris.target.shape
print "\n"

import numpy as np
print "(iris) target: unique items:"
print np.unique(iris.target)
print "\n"

digits = datasets.load_digits()
print "Digits: (N, pixelRows, pixelCols)"
print digits.images.shape
print "\n"

import pylab as pl
print "picture (8*8 pixels):"
print pl.imshow(digits.images[0], cmap=pl.cm.gray_r)
print "\n"

data = digits.images.reshape((digits.images.shape[0], -1))
print "transforming vector as 8*8 image:"
print data
print "\n"


from sklearn import svm
clf = svm.LinearSVC()
print "svm"
print clf.fit(iris.data, iris.target)
print "\n"

clf.predict([[ 5.0,  3.6,  1.3,  0.25]])
print "coef"
print clf.coef_   
