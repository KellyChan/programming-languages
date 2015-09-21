#-----------------------------------------------------------#
# Project: Pipeline Anova SVM
# Author: Kelly Chan
# Date: Apr 22 2014
#-----------------------------------------------------------#

print(__doc__)

from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline


def loadData():

    # generating data
    X, y = samples_generator.make_classification(n_features=20, \
                                                 n_informative=3, \
                                                 n_redundant=0, \
                                                 n_classes=4, \
                                                 n_clusters_per_class=2)
    return X, y


# ANOVA SVM-C
def createANOVASVM():
    
    # anova filter, take 3 best ranked features
    anova_filter = SelectKBest(f_regression, k=3)   
    # svm
    clf = svm.SVC(kernel='linear')

    anova_svm = Pipeline([('anova', anova_filter), \
                          ('svm', clf)])

    return anova_svm


def predict(X, y, anova_svm):
    anova_svm.fit(X, y)
    target = anova_svm.predict(X)
    return target 


def test():
    X, y = loadData()
    anova_svm = createANOVASVM()
    target = predict(X, y, anova_svm)
    print target


if __name__ == '__main__':
    test()


