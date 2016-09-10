import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class MarshOrchid(object):

    def __init__(self, filename):
        self.image = mpimg.imread(filename)
        self.height, self.width, self.depth = self.image.shape
    
    
    def transpose(self):
        x = tf.Variable(self.image, name='x')
        model = tf.initialize_all_variables()

        with tf.Session() as session:
            x = tf.transpose(x, perm=[1,0,2])
            session.run(model)
            result = session.run(x)
        return result


    def reverse(self):
        x = tf.Variable(self.image, name='x')
        model = tf.initialize_all_variables()
        
        with tf.Session() as session:
            x = tf.reverse_sequence(x, [self.width] * self.height, 1, batch_dim=0)
            session.run(model)
            result = session.run(x)
        return result


    def imshow(self, img):
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':

    marsh_orchid = MarshOrchid("./data/MarshOrchid.jpg")

    result = marsh_orchid.transpose()
    marsh_orchid.imshow(result)

    result = marsh_orchid.reverse()
    marsh_orchid.imshow(result)
