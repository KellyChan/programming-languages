import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class SoftMax(object):

    def __init__(self):
        self.x = tf.placeholder("float")
        self.y = tf.placeholder("float")
        self.W = tf.Variable([1.0, 2.0], name='w')


    def softmax(self):
        y_ = tf.mul(self.x, self.W[0]) + self.W[1]
        return y_


    def error(self, y_):
        error = tf.square(self.y - y_)
        return error

    def GDO(self, error):
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)
        return train_op


class SoftMaxTests(object):

    def __init__(self):
        self.softmax = SoftMax()

    def test1(self):
        y_ = self.softmax.softmax()
        error = self.softmax.error(y_)
        train_op = self.softmax.GDO(error)
        
        model = tf.initialize_all_variables()
        with tf.Session() as session:
            session.run(model)
            for i in range(1000):
                x_value = np.random.rand()
                y_value = x_value * 2 + 6
                session.run(train_op, feed_dict={self.softmax.x: x_value, self.softmax.y: y_value})
            w_value = session.run(self.softmax.W)
        print("Predicted model: {a: .3f}x + {b: .3f}".format(a=w_value[0], b=w_value[1])) 


    def test2(self):
        y_ = self.softmax.softmax()
        error = self.softmax.error(y_)
        train_op = self.softmax.GDO(error)
       
        errors = [] 
        model = tf.initialize_all_variables()
        with tf.Session() as session:
            session.run(model)
            for i in range(1000):
                x_train = tf.random_normal((1,), mean=5, stddev=2.0)
                y_train = x_train * 2 + 6
                x_value, y_value = session.run([x_train, y_train])
                _, error_value = session.run([train_op, error], feed_dict={self.softmax.x: x_value, self.softmax.y: y_value})
                print("Index: %d, Error: %f" % (i, error_value))
                errors.append(error_value)
            w_value = session.run(self.softmax.W)
        print("Predicted model: {a: .3f}x + {b: .3f}".format(a=w_value[0], b=w_value[1])) 
        return errors


    def plot(self, errors):
        plt.plot([np.mean(errors[i-50:i]) for i in range(len(errors))])
        plt.show()
        plt.savefig("./tmp/errors.png")
        

if __name__ == '__main__':

    softmax_test = SoftMaxTests()
    softmax_test.test1()
    errors = softmax_test.test2()
    softmax_test.plot(errors)
