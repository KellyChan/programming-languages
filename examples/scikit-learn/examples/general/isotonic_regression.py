#-----------------------------------------------------------#
# Project: Precision Recall
# Author: Kelly Chan
# Date: Apr 22 2014
#-----------------------------------------------------------#

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state


def loadData():
    n = 100
    x = np.arange(n)
    rs = check_random_state(0)
    y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
    return x, y


def createIsotonicRegression():
    ir = IsotonicRegression()
    return ir

def createLinearRegression():
    lr = LinearRegression()
    return lr

def predictIsotonicRegression(ir, x, y):
    return ir.fit_transform(x, y)

def predictLinearRegression(lr, x, y):
    return lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression

def plotRegression(x, y, y_, lr):

    segements = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
    lc = LineCollection(segements, zorder=0)
    lc.set_array(np.ones(len(y)))
    lc.set_linewidths(0.5 * np.ones(n))

    fig = plt.figure()
    plt.plot(x, y, 'r.', markersize=12)
    plt.plot(x, y_, 'g.-', markersize=12)
    plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
    plt.gca().add_collection(lc)
    plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
    plt.title('Isotonic regression')
    plt.show()


def test():
    x, y = loadData()
    ir = createIsotonicRegression()
    lr = createLinearRegression()
    y_ir = predictIsotonicRegression(ir, x, y)
    y_lr = predictIsotonicRegression(lr, x, y)
    plotRegression(x, y, y_ir, lr)

if __name__ == '__main__':
    test()


