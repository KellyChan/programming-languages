"""
Distances

- Manhattan (L1 distance): d1(I1, I2) = sum( I1 - I2 )
- Euclidean (L2 distance): d2(I1, I2) = sqrt( sum((I1 - I2)^2) )
"""

import time
import numpy as np


class Distances(object):

    def __init__(self, X1, X2):
        self.X1 = X1
        self.X2 = X2

    
    def manhattan_two_loops(self):
        X1_rows = self.X1.shape[0]
        X2_rows = self.X2.shape[0]
        distances = np.zeros((X1_rows, X2_rows))

        for x1_index in xrange(X1_rows):
            for x2_index in xrange(X2_rows):
                distances[x1_index, x2_index] = np.sum(np.abs(self.X1[x1_index, :] - self.X2[x2_index, :]))
        return distances

    def manhattan_one_loop(self):
        X1_rows = self.X1.shape[0]
        X2_rows = self.X2.shape[0]
        distances = np.zeros((X1_rows, X2_rows))

        for index in xrange(X1_rows):
            distances[index, :] = np.sum(np.abs(self.X1[index, :] - self.X2), axis=1)
        return distances


    def manhattan_no_loop(self):
        X1_rows = self.X1.shape[0]
        X2_rows = self.X2.shape[0]
        distances = np.zeros((X1_rows, X2_rows))
        return distances


    def euclidean_two_loops(self):
        X1_rows = self.X1.shape[0]
        X2_rows = self.X2.shape[0]
        distances = np.zeros((X1_rows, X2_rows))
        
        for x1_index in xrange(X1_rows):
            for x2_index in xrange(X2_rows):
                distances[x1_index, x2_index] = np.sqrt(np.sum((self.X1[x1_index, :] - self.X2[x2_index, :]) ** 2))
        return distances

 
    def euclidean_one_loop(self):
        X1_rows = self.X1.shape[0]
        X2_rows = self.X2.shape[0]
        distances = np.zeros((X1_rows, X2_rows))

        for index in xrange(X1_rows):
            distances[index, :] = np.sqrt(np.sum(np.square(self.X2 - self.X1[index, :]), axis=1))  # broadcasting

        return distances


    def euclidean_no_loop(self):
        X1_rows = self.X1.shape[0]
        X2_rows = self.X2.shape[0]
        distances = np.zeros((X1_rows, X2_rows))

        X1_sum = np.sum(np.square(self.X1), axis=1)
        X2_sum = np.sum(np.square(self.X2), axis=1)
        inner_product = np.dot(self.X1, self.X2.T)
        distances = np.sqrt( -2 * inner_product + X1_sum.reshape(-1, 1) + X2_sum)

        return distances


    def eval_distances(self, distances1, distances2):
        difference = np.linalg.norm(distances1 - distances2, ord='fro')

        print "Difference was: %f" % (difference, )
        if difference < 0.001:
            print "Good! The distance matrices are the same."
        else:
            print "Uh-oh! The distance matrices was different."

        return difference


    def eval_time(self, f, *args):
        """
        Call a function f with *args and return the time (in seconds) that it took to execute.
        """

        tic = time.time()
        f(*args)
        toc = time.time()

        cost = toc - tic
        print "%s: took %f seconds" % (f, cost)
        return cost
