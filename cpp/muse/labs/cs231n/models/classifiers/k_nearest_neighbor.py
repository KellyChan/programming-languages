import numpy as np

from models.algorithms.distances import Distances


class KNearestNeighbor(object):
    """ A KNN classifier with L2 distance. """

    def __init_(self):
        pass


    def train(self, X_train, Y_train):
        """"""

        self.X_train = X_train
        self.Y_train = Y_train

   
    def predict(self, X_test, k=1, distance_type='euclidean_no_loop'):
        distances = Distances(X_test, self.X_train)
        if distance_type == 'manhattan_two_loops':
            this_distances = distances.manhattan_two_loops()
        elif distance_type == 'manhattan_one_loop':
            this_distances = distances.manhattan_one_loop()
        elif distance_type == 'manhattan_no_loop':
            this_distances = distances.manhattan_no_loop()
        elif distance_type == 'euclidean_two_loops':
            this_distances = distances.euclidean_two_loops()
        elif distance_type == 'euclidean_one_loop':
            this_distances = distances.euclidean_one_loop()
        elif distance_type == 'euclidean_no_loop':
            this_distances = distances.euclidean_no_loop()
        else:
            raise ValueError("Invalid type %s for distance_type" % distance_type)

        return self._select_labels(this_distances, k=k)


    def _select_labels(self, distances, k=1):
        """"""

        num_of_test = distances.shape[0]
        Y_test_predictions = np.zeros(num_of_test)

        for test_index in xrange(num_of_test):
            # find the k nearest labels
            index_sorted = np.argsort(distances[test_index])
            y_nearest = self.Y_train[index_sorted]
            k_nearest = y_nearest[:k]

            # find the most frequency labels
            k_nearest = list(k_nearest)
            k_nearest_sorted = sorted(k_nearest, key=k_nearest.count, reverse=True)
            Y_test_predictions[test_index] = k_nearest_sorted[0] 

        return Y_test_predictions


    def eval_accuracy(self, Y_test, Y_test_predictions):
        num_of_test = len(Y_test)
        corrections = np.sum(Y_test == Y_test_predictions)
        accuracy = float(corrections) / num_of_test
        print "Got %d / %d correct => accuracy: %f" % (corrections, num_of_test, accuracy)
        return accuracy


    def eval_distances(self, X_test):
        distances = Distances(X_test, self.X_train)

        distances_manhattan_two_loops = distances.manhattan_two_loops()
        distances_manhattan_one_loop = distances.manhattan_one_loop()
        distances_manhattan_no_loop = distances.manhattan_no_loop()
        difference = distances.eval_distances(distances_manhattan_two_loops, distances_manhattan_one_loop)
        difference = distances.eval_distances(distances_manhattan_two_loops, distances_manhattan_no_loop)

        distances_euclidean_two_loops = distances.euclidean_two_loops()
        distances_euclidean_one_loop = distances.euclidean_one_loop()
        distances_euclidean_no_loop = distances.euclidean_no_loop()
        difference = distances.eval_distances(distances_euclidean_two_loops, distances_euclidean_one_loop)
        difference = distances.eval_distances(distances_euclidean_two_loops, distances_euclidean_no_loop)


    def eval_time(self, X_test):
        distances = Distances(X_test, self.X_train)

        distances.eval_time(distances.manhattan_two_loops)
        distances.eval_time(distances.manhattan_one_loop)
        distances.eval_time(distances.manhattan_no_loop)

        distances.eval_time(distances.euclidean_two_loops)
        distances.eval_time(distances.euclidean_one_loop)
        distances.eval_time(distances.euclidean_no_loop)


    def cross_validate(self, folds, k_list):
        """
        return k_to_accuracies:
            ::
                {
                   <fold>: [accuracy1, accuracy2, accuracy3, ...]
                }
        """

        X_train_folds = np.array_split(self.X_train, folds)
        Y_train_folds = np.array_split(self.Y_train, folds)

        k_to_accuracies = {}
        for k in k_list:
            k_to_accuracies[k] = []
            for fold in range(folds):
                this_X_train = np.vstack(X_train_folds[:fold]+X_train_folds[fold+1:])
                this_X_test = X_train_folds[fold]
                this_Y_train = np.hstack(Y_train_folds[:fold]+Y_train_folds[fold+1:])
                this_Y_test = Y_train_folds[fold]

                self.train(this_X_train, this_Y_train)
                Y_test_predictions = self.predict(this_X_test, k=k, distance_type='euclidean_no_loop')
                
                num_of_corrections = np.sum(Y_test_predictions == this_Y_test)
                accuracy = float(num_of_corrections) / len(this_Y_test)
                k_to_accuracies[k].append(accuracy)

        print "k_to_accuracies: \n ", k_to_accuracies
        # for k in sorted(k_to_accuracies):
        #    for accuracy in k_to_accuracies[k]:
        #        print 'k = %d, accuracy = %f' % (k, accuracy)

        return k_to_accuracies
