#!/usr/local/bin/python
import parser
import numpy as np
from sklearn import svm

# SVM solver for city classification problem
def svmSolver(svmType = "one-vs-the-rest"):
    
    # load data
    (IDs, cities, locations, ratings, featureVector) = parser.parseData()

    svmType = "one-against-one"

    # setup problem in terms of X,Y
    X = featureVector
    Y = []
    for city in cities:
        if city == "Phoenix":
            Y.append(0)
        elif city == "Las Vegas":
            Y.append(1)
        elif city == "Madison":
            Y.append(2)
        elif city == "Waterloo":
            Y.append(3)
        elif city == "Edinburgh":
            Y.append(4)
        else:
            print "Error: unrecognized city"

    testX = [np.ones(len(X[0]))]

    # setup one-against-one SVM
    if svmType == "one-against-one":
        clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,\
                gamma=0.0, kernel='rbf', max_iter=-1, probability=False, \
                random_state=None, shrinking=True, tol=0.001, verbose=False)
        clf.fit(X, Y)
        dec = clf.decision_function(testX)
        print dec

    # setup one-vs-the-rest
    elif svmType == "one-vs-the-rest":
        
        lin_clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True,\
                fit_intercept=True, intercept_scaling=1, loss='l2', \
                multi_class='ovr', penalty='l2', random_state=None, \
                tol=0.0001, verbose=0)
        lin_clf.fit(X, Y)
        dec = lin_clf.decision_function(testX)
        print dec

if __name__ == "__main__":
    svmSolver()       
