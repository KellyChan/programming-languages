import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class Conway(object):

    def __init__(self):
        self.shape = (50, 50)
        self.session = tf.Session()
 

    def run(self):
        initial_board = self.init_board()

        board = tf.placeholder(tf.int32, shape=self.shape, name='board')
        board_update = tf.py_func(self.update_board, [board], [tf.int32])

        initial_board_values = self.session.run(initial_board)
        X = self.session.run(board_update, feed_dict={board: initial_board_values})[0]

        ani = animation.FuncAnimation(self.plot(X), self.game_of_life, interval=200, blit=True)
        plt.show()

        return X


    def init_board(self):
        initial_board = tf.random_uniform(self.shape, minval=0, maxval=2, dtype=tf.int32)
        return initial_board


    def update_board(self, X):
        N = convolve2d(X, np.ones((3,3)), mode='same', boundary='wrap') - X
        X = (N == 3) | (X & (N == 2))
        return X


    def game_of_life(self, *args):
        X = self.session.run(board_update, feed_dict={board: X})[0]
        plot.set_array(X)
        return plot,


    def plot(self, X):
        fig = plt.figure()
        plot = plt.imshow(X, cmap='Greys', interpolation='nearest')
        plt.show()
        return fig


if __name__ == '__main__':

    conway = Conway()
    X = conway.run()
