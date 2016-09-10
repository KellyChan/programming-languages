#---------------------------------------------------------------#
# Project: Explicit feature map approximation for RBF kernels
# Author: Kelly Chan
# Date: Apr 25 2014
#---------------------------------------------------------------#

print(__doc__)

import numpy as np
import pylab as pl
from time import time

from sklearn import datasets
from sklearn.decomposition import PCA

from sklearn import svm
from sklearn import pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import Nystroem


def loadData():

    digits = datasets.load_digits(n_class=9)
    
    # flatten images, to turn data in a matrix (samples, feature)
    n_samples = len(digits.data)
    data = digits.data / 16.
    data -= data.mean(axis=0)

    data_train = data[:n_samples/2] 
    targets_train = digits.target[:n_samples/2]

    return data_train, targets_train



def createSVM():
    kernel_svm = svm.SVC(gamma=.2)
    linear_svm = svm.LinearSVC()
    return kernel_svm, linear_svm

def createPipeline():
    feature_map_fourier = RBFSampler(gamma=.2, random_state=1)
    feature_map_nystroem = Nystroem(gamma=.2, random_state=1)
    fourier_approx_svm = pipeline.Pipeline([('feature_map', feature_map_fourier), \
                                            ('svm', svm.LinarSVC())])
    nystroem_approx_svm = pipeline.Pipeline([('feature_map', feature_map_nystroem), \
                                             ('svm', svm.LinearSVC())])
    return fourier_approx_svm, nystroem_approx_svm

def predict(data_train, targets_train):
    
    kernel_svm_time = time()
    kernel_svm.fit(data_train, targets_train)
    kernel_svm_score = kernel_svm.score(data_test, targets_test)
    kernel_svm_time = time() - kernel_svm_time

    linear_svm_time = time()
    linear_svm.fit(data_train, targets_train)
    linear_svm_score = linear_svm.score(data_test, targets_test)
    linear_svm_time = time() - linear_svm_time

    sample_sizes = 30 * np.arange(1, 10)
    fourier_scores = []
    nystroem_scores = []
    fourier_times = []
    nystroem_times = []

    for D in sample_sizes:
        fourier_approx_svm.set_params(feature_map__n_components=D)
        nystroem_approx_svm.set_params(feature_map__n_components=D)
        start = time()
        nystroem_approx_svm.fit(data_train, targets_train)
        nystroem_times.append(time() - start)

        start = time()
        fourier_approx_svm.fit(data_train, targets_train)
        fourier_times.append(time() - start)

        fourier_score = fourier_approx_svm.score(data_test, targets_test)
        nystroem_score = nystroem_approx_svm.score(data_test, targets_test)
        nystroem_scores.append(nystroem_score)
        fourier_scores.append(fourier_score)


def plotResults():

    pl.figure(figsize=(8, 8))

    accuracy = pl.subplot(211)
    timescale = pl.subplot(212)

    accuracy.plot(sample_sizes, \
                  nystroem_scores, \
                  label="Nystroem approx. kernel")

    timescale.plot(sample_sizes, \
                   nystroem_times, \
                   '--', \
                   label='Nystroem approx. kernel')

    accuracy.plot(sample_sizes, fourier_scores, label="Fourier approx. kernel")
    timescale.plot(sample_sizes, fourier_times, '--',
               label='Fourier approx. kernel')

    # horizontal lines for exact rbf and linear kernels:
    accuracy.plot([sample_sizes[0], sample_sizes[-1]],
                  [linear_svm_score, linear_svm_score], label="linear svm")
    timescale.plot([sample_sizes[0], sample_sizes[-1]],
                   [linear_svm_time, linear_svm_time], '--', label='linear svm')

    accuracy.plot([sample_sizes[0], sample_sizes[-1]],
                  [kernel_svm_score, kernel_svm_score], label="rbf svm")
    timescale.plot([sample_sizes[0], sample_sizes[-1]],
                   [kernel_svm_time, kernel_svm_time], '--', label='rbf svm')

    # vertical line for dataset dimensionality = 64
    accuracy.plot([64, 64], [0.7, 1], label="n_features")

    # legends and labels
    accuracy.set_title("Classification accuracy")
    timescale.set_title("Training times")
    accuracy.set_xlim(sample_sizes[0], sample_sizes[-1])
    accuracy.set_xticks(())
    accuracy.set_ylim(np.min(fourier_scores), 1)
    timescale.set_xlabel("Sampling steps = transformed feature dimension")
    accuracy.set_ylabel("Classification accuracy")
    timescale.set_ylabel("Training time in seconds")
    accuracy.legend(loc='best')
    timescale.legend(loc='best')

    # visualize the decision surface, projected down to the first
    # two principal components of the dataset
    pca = PCA(n_components=8).fit(data_train)

    X = pca.transform(data_train)

    # Gemerate grid along first two principal components
    multiples = np.arange(-2, 2, 0.1)
    # steps along first component
    first = multiples[:, np.newaxis] * pca.components_[0, :]
    # steps along second component
    second = multiples[:, np.newaxis] * pca.components_[1, :]
    # combine
    grid = first[np.newaxis, :, :] + second[:, np.newaxis, :]
    flat_grid = grid.reshape(-1, data.shape[1])

    # title for the plots
    titles = ['SVC with rbf kernel',
              'SVC (linear kernel)\n with Fourier rbf feature map\n'
              'n_components=100',
              'SVC (linear kernel)\n with Nystroem rbf feature map\n'
              'n_components=100']

    pl.tight_layout()
    pl.figure(figsize=(12, 5))

    # predict and plot
    for i, clf in enumerate((kernel_svm, nystroem_approx_svm,
                             fourier_approx_svm)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        pl.subplot(1, 3, i + 1)
        Z = clf.predict(flat_grid)

        # Put the result into a color plot
        Z = Z.reshape(grid.shape[:-1])
        pl.contourf(multiples, multiples, Z, cmap=pl.cm.Paired)
        pl.axis('off')

        # Plot also the training points
        pl.scatter(X[:, 0], X[:, 1], c=targets_train, cmap=pl.cm.Paired)

        pl.title(titles[i])
    pl.tight_layout()
    pl.show()
