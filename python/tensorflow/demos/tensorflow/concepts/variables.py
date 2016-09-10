import numpy as np
import tensorflow as tf


class Plus(object):

    def __init__(self):
        self.x = 0


    def plus1(self):
        x = tf.constant(35, name='x')
        y = tf.Variable(x+5, name='y')
        model = tf.initialize_all_variables()

        with tf.Session() as session:
            session.run(model)
            print(session.run(y))


    def plus2(self):
        x = tf.constant([35, 40, 45], name='x')
        y = tf.Variable(x+5, name='y')
        model = tf.initialize_all_variables()

        with tf.Session() as session:
            session.run(model)
            print(session.run(y))


    def plus3(self):
        data = np.random.randint(1000, size=10000)
        x = tf.constant(data, name='x')
        y = tf.Variable(x+5, name='y')
        model = tf.initialize_all_variables()

        with tf.Session() as session:
            session.run(model)
            print(session.run(y))


    def plus4(self):
        x = tf.Variable(0, name='x')
        model = tf.initialize_all_variables()

        with tf.Session() as session:
            for i in range(5):
                session.run(model)
                x = x+1
                print(session.run(x))


    def plus5(self):
        x = tf.constant(35, name='x')
        print(x)
        y = tf.Variable(x+5, name='y')

        with tf.Session() as session:
            merged = tf.merge_all_summaries()
            writer = tf.train.SummaryWriter("./tmp/basic", session.graph_def)
            model = tf.initialize_all_variables()
            session.run(model)
            print(session.run(y))


if __name__ == '__main__':

    plus = Plus()
    plus.plus1()
    plus.plus2()
    plus.plus3()
    plus.plus4()
    plus.plus5()
