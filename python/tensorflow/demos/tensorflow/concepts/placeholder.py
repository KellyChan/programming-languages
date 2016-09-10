import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class MarshOrchid(object):


    def __init__(self, filename):
        self.image = mpimg.imread(filename)


    def placeholder1(self):
        x = tf.placeholder("float", 3)
        y = x * 2
        with tf.Session() as session:
            result = session.run(y, feed_dict={x: [1,2,3]})
            print(result)
        return result


    def placeholder2(self):
        x = tf.placeholder("float", [None, 3])
        y = x * 2
        with tf.Session() as session:
            x_data = [[1,2,3], [4,5,6]]
            result = session.run(y, feed_dict={x: x_data})
            print(result)
        return result


    def placeholder3(self):
        image = tf.placeholder("uint8", [None, None, 3])   
        slices = tf.slice(image, [1000, 0, 0], [3000, -1, -1])
        with tf.Session() as session:
            result = session.run(slices, feed_dict={image: self.image}) 
            print(result.shape)
        return result


    def imshow(self, img=''):
        if img == '':
            plt.imshow(self.image)
        else:
            plt.imshow(img)
        plt.show()


if __name__ == '__main__':

    marsh_orchid = MarshOrchid("./data/MarshOrchid.jpg")
    marsh_orchid.imshow('')

    result = marsh_orchid.placeholder1()
    result = marsh_orchid.placeholder2()

    result = marsh_orchid.placeholder3()
    marsh_orchid.imshow(result)
