import numpy as np
import matplotlib.pyplot as plt

import settings
from data.datasets import CIFAR10
from models.classifiers.k_nearest_neighbor import KNearestNeighbor


def visualize(data, classes=settings.CIFAR10_CLASSES, samples_per_class=7):

    total_classes = len(classes)

    for index, cifar10_class in enumerate(classes):
        indexes = np.flatnonzero(data[1] == index)
        sample_indexes = np.random.choice(indexes, samples_per_class, replace=False)
        for i, sample_index in enumerate(sample_indexes):
            plt_index = i * total_classes + index + 1
            plt.subplot(samples_per_class, total_classes, plt_index)
            plt.imshow(data[0][sample_index].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cifar10_class)
    plt.show()
    

def sampling(data, quota):
    mask = range(quota)
    return data[mask]


def predict_with_distances(X_train, Y_train, X_test, Y_test):

    classifier = KNearestNeighbor()
    classifier.train(X_train, Y_train)

    Y_test_predictions = classifier.predict(X_test, k=1, distance_type='manhattan_two_loops')
    classifier.eval_accuracy(Y_test, Y_test_predictions)

    Y_test_predictions = classifier.predict(X_test, k=1, distance_type='manhattan_one_loop')
    classifier.eval_accuracy(Y_test, Y_test_predictions)

    Y_test_predictions = classifier.predict(X_test, k=1, distance_type='manhattan_no_loop')
    classifier.eval_accuracy(Y_test, Y_test_predictions)

    Y_test_predictions = classifier.predict(X_test, k=1, distance_type='euclidean_two_loops')
    classifier.eval_accuracy(Y_test, Y_test_predictions)

    Y_test_predictions = classifier.predict(X_test, k=1, distance_type='euclidean_one_loop')
    classifier.eval_accuracy(Y_test, Y_test_predictions)

    Y_test_predictions = classifier.predict(X_test, k=1, distance_type='euclidean_no_loop')
    classifier.eval_accuracy(Y_test, Y_test_predictions)


def compare_distances(X_train, Y_train, X_test):
    classifier = KNearestNeighbor()
    classifier.train(X_train, Y_train)

    classifier.eval_distances(X_test)


def compare_time(X_train, Y_train, X_test):
    classifier = KNearestNeighbor()
    classifier.train(X_train, Y_train)

    classifier.eval_time(X_test)


def cross_validate(X_train, Y_train, folds, k_list):
    classifier = KNearestNeighbor()
    classifier.train(X_train, Y_train)

    k_to_accuracies = classifier.cross_validate(folds, k_list)
    for k in k_list:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_list, accuracies_mean, yerr=accuracies_std)
    plt.title("Cross-Validation on k")
    plt.xlabel('k')
    plt.ylabel('Cross-Validation accuracy')
    plt.show()


def k_classifier(X_train, Y_train, X_test, Y_test, k):
    classifier = KNearestNeighbor()
    classifier.train(X_train, Y_train)
    Y_test_predictions = classifier.predict(X_test, k=k, distance_type='euclidean_no_loop')
    accuracy = classifier.eval_accuracy(Y_test, Y_test_predictions)
    return [accuracy, Y_test_predictions]


def main():
    """"""

    # load data
    cifar10 = CIFAR10(settings.CIFAR10)
    data_train = cifar10.load_train('data_batch_%d')
    data_test = cifar10.load_test('test_batch')
    print data_train[0].shape
    print data_train[1].shape
    print data_test[0].shape
    print data_test[1].shape
    # visualize(data=data_train)

    # sampling
    X_train = sampling(data_train[0], 5000)
    Y_train = sampling(data_train[1], 5000)
    X_test = sampling(data_test[0], 500)
    Y_test = sampling(data_test[1], 500)

    # reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print X_train.shape, X_test.shape

    # train
    # predict_with_distances(X_train, Y_train, X_test, Y_test)
    # compare_distances(X_train, Y_train, X_test)
    # compare_time(X_train, Y_train, X_test)

    # cross validation
    # folds = 5
    # k_list = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    # cross_validate(X_train, Y_train, folds, k_list)
    
    # k_train
    best_k = 8
    k_classifier(X_train, Y_train, X_test, Y_test, best_k)

    # distances = classifier.compute_distances_two_loops(X_test)
    # print distances.shape
    # plt.imshow(distances, interpolation='none')
    # plt.show()

    # predict
    # Y_test_predictions = classifier.predict(distances, k=1)
    # accuracy = classifier.evaluate(Y_test, Y_test_predictions)
    # Y_test_predictions = classifier.predict(distances, k=5)
    # accuracy = classifier.evaluate(Y_test, Y_test_predictions)

    # distance_one_loop
    # distances_one = classifier.compute_distances_one_loop(X_test)
    # print distances_one.shape
    # difference = classifier.evaluate_distances(distances, distances_one)



if __name__ == '__main__':

    main()
