import tensorflow as tf

class Convergence(object):

    def __init__(self):
        self.x = 0

    def loop1(self):
        x = tf.Variable(0, name='x')
        model = tf.initialize_all_variables()
        with tf.Session() as session:
            for i in range(5):
                session.run(model)
                x = x + 1
                print(session.run(x))


    def loop2(self):
        x = tf.Variable(0., name='x')
        threshold = tf.constant(5.)
        model = tf.initialize_all_variables()
        with tf.Session() as session:
            session.run(model)
            while session.run(tf.less(x, threshold)):
                x = x + 1
                x_value = session.run(x)
                print(x_value)


if __name__ == '__main__':

    convergence = Convergence()
   
    convergence.loop1()
    convergence.loop2()
