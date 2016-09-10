#---------------------------------------------------------------#
# Project: Sample pipeline for text feature extraction and evaluation
# Author: Kelly Chan
# Date: Apr 23 2014
#---------------------------------------------------------------#

from __future__ import print_function

from pprint import pprint
from time import time
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

print(__doc__)

# display progress logs on stdout
logging.basicConfig(level=logging.INFO, \
                    format='%(asctime)s %(levelname)s %(message)s')

def loadCategories():

    # load some categories from the training set
    categories = ['alt.atheism', \
                  'talk.religion.misc'
                 ]

    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    return categories

def loadData(categories):
    
    data = fetch_20newsgroups(subset='train', \
                              categories=categories)

    print("%d documents" % len(data.filenames))
    print("%d categories" % len(data.target_names))
    print()

    return data

def createPipeline():

    # define a pipeline combining a text feature extractor
    pipeline = Pipeline([('vect', CountVectorizer()), \
                         ('tfidf', TfidfTransformer()), \
                         ('clf', SGDClassifier()) \
                        ])
    return pipeline


def defineParameters():
    parameters = {
                  'vect__max_df': (0.5, 0.75, 1.0), \
                  #'vect__max_features': (None, 5000, 10000, 50000), \
                  'vect__ngram_range': ((1, 1), (1, 2)), \
                  #'tfidf__use_idf': (True, False),
                  #'tfidf__norm': ('l1', 'l2'),
                  'clf__alpha': (0.0001, 0.000001), \
                  'clf__penalty': ('12', 'elasticnet'), \
                  #'clf__n_iter': (10, 50, 80), \
                 }

    return parameters


if __name__ == "__main__":

    categories = loadCategories()
    data = loadData(categories)

    pipeline = createPipeline()
    parameters = defineParameters()

    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)


    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %.3fs" % (time() - t0))
    print()


    print("Best score: %.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


