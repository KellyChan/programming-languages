#-----------------------------------------------------------#
# Project: Concatenating multiple feature extraction methods
# Author: Kelly Chan
# Date: Apr 22 2014
#-----------------------------------------------------------#


print(__doc__)

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest


def loadData():
    iris = load_iris()
    X, y = iris.data, iris.target
    return X, y

def computePCA():
    pca = PCA(n_components=2)
    return pca

def selectFeatures():
    selection = SelectKBest(k=1)
    return selection

def combineFeatures(pca, selection):
    combined_features = FeatureUnion([("pca", pca), \
                                      ("univ_select", selection)])  # Univariate selection
    return combined_features

def transformData(combined_features, X, y):
    X_features = combined_features.fit(X, y).transform(X)
    return X_features

def createSVM():
    svm = SVC(kernel='linear')
    return svm

def classify(svm, X_features, y):
    return svm.fit(X_features, y)

def searchGrid(combined_features, svm, X, y):
    pipeline = Pipeline([('features', combined_features), \
                         ('svm', svm)])

    param_grid = dict(features__pca__n_components=[1, 2, 3], \
                      features__univ_select__k=[1, 2], \
                      svm__C=[0.1, 1, 10])

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
    grid_search.fit(X, y)
    print(grid_search.best_estimator_)
    return grid_search.best_estimator_


def test():

    X, y = loadData()

    pca = computePCA()
    selection = selectFeatures
    combined_features = combineFeatures(pca, selection)

    svm = createSVM()
    X_features = transformData(combined_features, X, y)
    #y_pred = classify(svm, X_features, y)

    #searchGrid(combined_features, svm, X, y)


if __name__ == '__main__':
    test()


