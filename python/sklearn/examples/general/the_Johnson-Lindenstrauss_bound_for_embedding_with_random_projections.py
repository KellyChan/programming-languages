#---------------------------------------------------------------#
# Project: The Johnson-Lindenstrauss bound for embedding with random projections
# Author: Kelly Chan
# Date: Apr 24 2014
#---------------------------------------------------------------#

print(__doc__)

import sys
from time import time

import numpy as np
import pylab as pl

from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import euclidean_distances


def plotDependencyComponents():

    """Plot thoretical dependency between n_samples and n_components"""

    # range of admissible distortions
    eps_range = np.linspace(0.1, 0.99, 5)
    colors = pl.cm.Blues(np.linspace(0.3, 1.0, len(eps_range)))

    # range of number of samples to embed
    n_samples_range = np.logspace(1, 9, 9)

    
    pl.figure()

    for eps, color in zip(eps_range, colors):
        min_n_components = johnson_lindenstrauss_min_dim(n_samples_range, \
                                                         eps=eps)
        pl.loglog(n_samples_range, min_n_components, color=color)

    pl.legend(["eps = %.1f" % eps for eps in eps_range], \
              loc="lower right")

    pl.xlabel("Number of observations to eps-embed")
    pl.ylabel("Minimum number of dimensions")
    pl.title("Johnson-Lindenstrauss bounds:\nn_samples vs n_components")
    pl.show()


def plotDependencyEPS():

    """Plot thoretical dependency between n_components and eps"""
    
    # range of admissible distortions
    eps_range = np.linspace(0.01, 0.99, 100)

    # range of number of samples to embed
    n_samples_range = np.logspace(2, 6, 5)
    colors = pl.cm.Blues(np.linspace(0.3, 1.0, len(n_samples_range)))

    pl.figure()

    for n_samples, color in zip(n_samples_range, colors):
        min_n_components = johnson_lindenstrauss_min_dim(n_samples, \
                                                         eps=eps_range)
        pl.semilogy(eps_range, min_n_components, color=color)

    pl.legend(["n_samples = %d" % n for n in n_samples_range], \
              loc="upper right")

    pl.xlabel("Distortion eps")
    pl.ylabel("Minimum number of dimensions")
    pl.title("Johnson-Lindenstrauss bounds:\nn_components vs eps")
    pl.show()


def loadData():

    """digits images: low dimensional and dense"""
    """20 newsgroups: high dimensional and sparse"""
    
    if '--twenty-newsgroups' in sys.argv:
        # need an internet connection hence not enabled by default
        data = fetch_20newsgrouops_vectorized().data[:500]
    else:
        data = load_digits().data[:500]

    n_samples, n_features = data.shape
    print("Embedding %d samples with dim %d using various random projections" \
              % (n_samples, \
                 n_features))

    return data, n_samples, n_features

def plotHexbin(dists, projected_dists, n_components):

    pl.figure()

    pl.hexbin(dists, projected_dists, gridsize=100)
    pl.xlabel("Pairwise squared distances in original space")
    pl.ylabel("Pairwise squared distances in projected sapce")
    pl.title("Pairwise distances distribution for n_components=%d" \
                    % n_components)

    cb = pl.colorbar()
    cb.set_label('Sample pairs counts')


def plotHist(rates, n_components):
    
    pl.figure()

    pl.hist(rates, bins=50, normed=True, range=(0., 2.))
    pl.xlabel("Squared distances rate: projected / original")
    pl.ylabel("Distribution of samples pairs")
    pl.title("Histogram of pairwise distance rates for n_components=%d" \
                    % n_components)

    pl.show()


def plotProjection(data, n_samples, n_features): 

    n_components_range = np.array([300, 1000, 10000])
    dists = euclidean_distances(data, squared=True).ravel()

    # select only non-identical samples pairs
    nonzero = dists != 0
    dists = dists[nonzero]

    for n_components in n_components_range:

        t0 = time()

        rp = SparseRandomProjection(n_components=n_components)
        projected_data = rp.fit_transform(data)

        print("Projected %d samples from %d to %d in %.3fs" \
                % (n_samples, \
                   n_features, \
                   n_components, \
                   time() - t0))

        if hasattr(rp, 'components_'):
            n_bytes = rp.components_.data.nbytes
            n_bytes += rp.components_.indices.nbytes
            print("Random matrix with size: %.3fMB" % (n_bytes / 1e6))


        projected_dists = euclidean_distances(projected_data, squared=True)
        projected_dists = projected_dists.ravel()[nonzero]

        rates = projected_dists / dists
        print("Mean distances rate: %.2f (%.2f)" \
                % (np.mean(rates), \
                   np.std(rates)))

        plotHexbin(dists, projected_dists, n_components)
        plotHist(rates, n_components)



if __name__ == '__main__':

    plotDependencyComponents()
    plotDependencyEPS()


    data, n_samples, n_features = loadData()
    plotProjection(data, n_samples, n_features)


