#---------------------------------------------------------------#
# Project: FeatureHasher and DictVectorizer Comparison
# Author: Kelly Chan
# Date: Apr 24 2014
#---------------------------------------------------------------#

from __future__ import print_function
from collections import defaultdict

import re
import sys
from time import time

import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

def n_nonzero_columns(X):
    """Returns the number of non-zero columns in a CSR matrix X."""
    return len(np.unique(X.nonzero()[1]))

def tokens(doc):
    """Extract tokens from doc.

    This uses a simple regex to break strings into tokens.
    For a more principled approach, see CountVectorizer or TfidfVectorizer.
    """
    return (tok.lower() for tok in re.findall(r"\w+", doc))

def token_freqs(doc):
    """Extract a dict mapping tokens from doc to their frequencies."""
    freq = defaultdict(int)
    for tok in tokens(doc):
        freq[tok] += 1
    return freq


def loadCategories():
    categories = [
                  'alt.atheism',
                  'comp.graphics',
                  'comp.sys.ibm.pc.hardware',
                  'misc.forsale',
                  'rec.autos',
                  'sci.space',
                  'talk.religion.misc',
                 ]
    return categories

def loadData(categories):

    raw_data = fetch_20newsgroups(subset='train', categories=categories).data
    data_size_mb = sum(len(s.encode('utf-8')) for s in raw_data) / 1e6

    print ("%d documents = %.3fMB" % (len(raw_data), data_size_mb))
    print()

    return raw_data, data_size_mb


def report(data_size_mb, termLens, t0):
    duration = time() - t0
    print("done in %fs at %.3fMB/s" % (duration, data_size_mb / duration))
    print("Found %d unique terms" % termLens)
    print()

def main():

    # Uncomment the following line to use a larger set (11k+ documents)
    #categories = None

    print(__doc__)
    print("Usage: %s [n_features_for_hashing]" % sys.argv[0])
    print("The default number of features is 2**18.")
    print()

    try:
        n_features = int(sys.argv[1])
    except IndexError:
        n_features = 2 ** 18
    except ValueError:
        print("not a valid number of features: %r" % sys.argv[1])
        sys.exit(1)

    print("Loading 20 newsgroups training data")
    categories = loadCategories()
    raw_data, data_size_mb = loadData(categories)

    
    print("DictVectorizer")
    t0 = time()
    vectorizer = DictVectorizer()
    vectorizer.fit_transform(token_freqs(d) for d in raw_data)
    report(data_size_mb, len(vectorizer.get_feature_names()), t0)


    print("FeatureHasher on frequency dicts")
    t0 = time()
    hasher = FeatureHasher(n_features=n_features)
    X = hasher.transform(token_freqs(d) for d in raw_data)
    report(data_size_mb, n_nonzero_columns(X), t0)


    print("FeatureHasher on raw tokens")
    t0 = time()
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    X = hasher.transform(tokens(d) for d in raw_data)
    report(data_size_mb, n_nonzero_columns(X), t0)


if __name__ == '__main__':
    main()
