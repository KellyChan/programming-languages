#---------------------------------------------------------------#
# Project: Clustering text documents using k-means
# Author: Kelly Chan
# Date: Apr 25 2014
#---------------------------------------------------------------#

from __future__ import print_function

import sys
import logging
from time import time
from optparse import OptionParser

import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans



# disply progress logs on stdout
logging.basicConfig(level=logging.INFO, \
                    format='%(asctime)s %(levelname)s %(message)s')




    
# parse commandline arguments
op = OptionParser()

op.add_option("--lsa", \
              dest="n_components", \
              type="int", \
              help="Preprocess documents with latent semantic analysis.")

op.add_option("--no-minibatch", \
              action="store_false", \
              dest="minibatch", \
              default=True, \
              help="Use ordinary k-means algorithm (in batch mode).")

op.add_option("--no-idf", \
              action="store_false", \
              dest="use_idf", \
              default=True, \
              help="Disable Inverse Document Frequency feature weighting.")

op.add_option("--use-hashing", \
              action="store_true", \
              default=False, \
              help="Use a hashing feature vectorizer")

op.add_option("--n-features", \
              type=int, \
              default=10000, \
              help="Maximum number of features (dimensions) to extract from text.")

op.add_option("--verbose", \
              action="store_true", \
              dest="verbose", \
              default=False, \
              help="Print progress reports inside k-means algorithm.")



(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()



def loadCategories():

    # load some categories from the training set
    categories = ['alt.atheism', \
                  'talk.religion.misc', \
                  'comp.graphics', \
                  'sci.space', \
                 ]
    return categories

def loadData(categories):

    # uncomment the following for all categories
    # categories = None

    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    dataset = fetch_20newsgroups(subset='all', \
                                 categories=categories, \
                                 shuffle=True, \
                                 random_state=42)

    print("%d documents" % len(dataset.data))
    print("%d categories" % len(dataset.target_names))
    print()

    return dataset



def extractFeatures(dataset):

    labels = dataset.target
    true_k = np.unique(labels).shape[0]

    print("Extracting features from the training dataset using a sparse vectorizer")

    t0 = time()
    if opts.use_hashing:
        if opts.use_idf:

            # perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=opts.n_features, \
                                       stop_words='english', \
                                       non_negative=True, \
                                       norm=None, \
                                       binary=False)

            vectorizer = Pipeline((('hasher', hasher), \
                                   ('tf_idf', TfidTransformer())))
        
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features, \
                                           stop_words='english', \
                                           non_negative=False, \
                                           norm='12', \
                                           binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, \
                                     max_features=opts.n_features, \
                                     stop_words='english', \
                                     use_idf=opts.use_idf)

    X = vectorizer.fit_transform(dataset.data)

    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()

    return X, labels



def reduceDimensions(X):
    
    print("Performing dimensionality reduction using LSA")

    t0 = time()

    lsa = TruncatedSVD(opts.n_components)
    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))
    print()

    return X

def normalize(X):

    print("Normalizing data for K-Means")

    t0 = time()
    X = Normalizer(copy=False).fit_transform(X)

    print("done in %fs" % (time() - t0))
    print()

    return X

def clusteringKMeans(X):

    if opts.minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, \
                             init='k-means++', \
                             n_init=1, \
                             init_size=1000, \
                             batch_size=1000, \
                             verbose=opts.verbose)
    else:
        km = KMeans(n_clusters=true_k, \
                    init='k-means++', \
                    max_iter=100, \
                    n_init=1, \
                    verbose=opts.verbose)

    print("Clustering sparse data with %s" % km)

    t0 = time()
    y_clust = km.fit(X)

    print("done in %.3fs" % (time() - t0))
    print()

    return y_clust, km


def evaluate(km, labels):

    print("Homogeneity: %.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %.3f" % metrics.v_measure_score(labels, km.labels_))

    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %.3f" % metrics.silhouette_score(X, \
                                                                    labels, \
                                                                    sample_size=1000))



def main():

    categories = loadCategories()
    dataset = loadData(categories)

    X, labels = extractFeatures(dataset)
    X = reduceDimensions(X)
    X = normalize(X)

    y_clust, km = clusteringKMeans(X)
    evaluate(km, labels)


if __name__ == '__main__':   
    
    main()


