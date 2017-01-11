import os
import numpy as np
import cPickle as pickle


class CIFAR10(object):
    """"""

    def __init__(self, data_dir):
        """"""

        self.data_dir = data_dir


    def load_train(self, filename_format):
        """"""

        X = []
        Y = []

        for batch_index in range(1, 6):  # batch 1-5
            batch_name = os.path.join(self.data_dir, filename_format % (batch_index, ))
            batch_X, batch_Y = self._load_batch(batch_name)
            X.append(batch_X)
            Y.append(batch_Y)
        X_train = np.concatenate(X)  # concatenate 5 batches of the dataset
        Y_train = np.concatenate(Y)
        del X, Y
     
        return [X_train, Y_train]


    def load_test(self, filename):
        """"""

        filename = os.path.join(self.data_dir, filename)
        X_test, Y_test = self._load_batch(filename)
        return [X_test, Y_test]


    def _load_batch(self, filename):
        """"""

        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
            X = data_dict['data']
            Y = data_dict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            return X, Y
