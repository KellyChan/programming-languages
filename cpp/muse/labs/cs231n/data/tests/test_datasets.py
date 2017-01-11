import unittest

from data.datasets import CIFAR10


class CIFAR10Tests(unittest.TestCase):


    def setUp(self):
        self.cifar10 = CIFAR10('/data/data/images/cifar10/cifar-10-batches-py')
   
 
    def test_load_train(self, filename_format='data_batch_%d'):
        data_train = self.cifar10.load_train(filename_format)
        self.assertEqual(2, len(data_train))
        self.assertEqual(50000, len(data_train[0]))
        self.assertEqual(50000, len(data_train[1]))
        self.assertEqual(32, len(data_train[0][0]))


    def test_load_test(self):
        data_test = self.cifar10.load_test('test_batch')
        self.assertEqual(2, len(data_test))
        self.assertEqual(10000, len(data_test[0]))
        self.assertEqual(10000, len(data_test[1]))
        self.assertEqual(32, len(data_test[0][0]))
