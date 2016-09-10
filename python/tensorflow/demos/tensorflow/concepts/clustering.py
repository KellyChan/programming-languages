import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class KMeans(object):

    def __init__(self, n_features, n_clusters, n_samples_per_cluster, seed, embiggen_factor):
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.n_samples_per_cluster = n_samples_per_cluster
        self.seed = seed
        self.embiggen_factor = embiggen_factor

    def train1(self):
        np.random.seed(self.seed)
        samples, centroids = self.create_samples(self.n_clusters,
                                                 self.n_samples_per_cluster,
                                                 self.n_features,
                                                 self.embiggen_factor,
                                                 self.seed)

        model = tf.initialize_all_variables()
        with tf.Session() as session:
            sample_values = session.run(samples)
            centroid_values = session.run(centroids)
        return sample_values, centroid_values


    def train2(self):
        np.random.seed(self.seed)
        samples, centroids = self.create_samples(self.n_clusters,
                                                 self.n_samples_per_cluster,
                                                 self.n_features,
                                                 self.embiggen_factor,
                                                 self.seed)
        initial_centroids = self.choose_random_centroids(samples, self.n_clusters)

        model = tf.initialize_all_variables()
        with tf.Session() as session:
            sample_values = session.run(samples)
            centroid_values = session.run(initial_centroids)
        return sample_values, centroid_values


    def train3(self):
        np.random.seed(self.seed)
        samples, centroids = self.create_samples(self.n_clusters,
                                                 self.n_samples_per_cluster,
                                                 self.n_features,
                                                 self.embiggen_factor,
                                                 self.seed)
        initial_centroids = self.choose_random_centroids(samples, self.n_clusters)
        nearest_indices = self.assign_to_nearest(samples, initial_centroids)
        updated_centroids = self.update_centroids(samples, nearest_indices, self.n_clusters)

        model = tf.initialize_all_variables()
        with tf.Session() as session:
            sample_values = session.run(samples)
            updated_centroid_values = session.run(updated_centroids)
            print(updated_centroid_values)
        return sample_values, updated_centroid_values



    def choose_random_centroids(self, samples, n_clusters):
        n_samples = tf.shape(samples)[0]
        random_indices = tf.random_shuffle(tf.range(0, n_samples))
        begin = [0,]
        size = [n_clusters,]
        size[0] = n_clusters
        centroid_indices = tf.slice(random_indices, begin, size)
        initial_centroids = tf.gather(samples, centroid_indices)
        return initial_centroids


    def assign_to_nearest(self, samples, centroids):
        expanded_vectors = tf.expand_dims(samples, 0)
        expanded_centroids = tf.expand_dims(centroids, 1)
        distances = tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroids)), 2)
        mins = tf.argmin(distances, 0)
        nearest_indices = mins
        return nearest_indices


    def update_centroids(self, samples, nearest_indices, n_clusters):
        nearest_indices = tf.to_int32(nearest_indices)
        partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
        new_centroids = tf.concat(0, [tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions])
        return new_centroids


    def create_samples(self, n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
        np.random.seed(seed)
        slices = []
        centroids = []
        for i in range(n_clusters):
            samples = tf.random_normal((n_samples_per_cluster, n_features),
                                       mean=0.0, stddev=5.0,
                                       dtype=tf.float32, seed=seed,
                                       name="cluster_{}".format(i))
            current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor/2)
            centroids.append(current_centroid)
            samples += current_centroid
            slices.append(samples)

        samples = tf.concat(0, slices, name='samples')
        centroids = tf.concat(0, centroids, name='centroids')
        return samples, centroids


    def plot_clusters(self, all_samples, centroids, n_samples_per_cluster):
        colour = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))
        for i, centroid in enumerate(centroids):
            samples = all_samples[i*n_samples_per_cluster:(i+1) * n_samples_per_cluster]
            plt.scatter(samples[:,0], samples[:,1], c=colour[i])
            plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
            plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
        plt.show()




if __name__ == '__main__':


    n_features = 2
    n_clusters = 3
    n_samples_per_cluster = 500
    seed = 700
    embiggen_factor = 70
    kmeans = KMeans(2, 3, 500, 700, 70)

    sample_values, centroid_values = kmeans.train1()
    kmeans.plot_clusters(sample_values, centroid_values, n_samples_per_cluster)

    sample_values, centroid_values = kmeans.train2()
    kmeans.plot_clusters(sample_values, centroid_values, n_samples_per_cluster)

    sample_values, centroid_values = kmeans.train3()
    kmeans.plot_clusters(sample_values, centroid_values, n_samples_per_cluster)

