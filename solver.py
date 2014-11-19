#!/usr/local/bin/python
import parser
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cluster import KMeans

# SVM solver for city classification problem
def classify(X,Y):

    # define SVM type
    svmType = "one-against-one"

    # form train, validation, and test datasets
    trainSize = len(X)/2
    valSize = len(X)/4
    testSize = len(X) - len(X)/2 - len(X)/4

    trainX = X[0:trainSize-1]
    trainY = Y[0:trainSize-1]
    valX = X[trainSize:trainSize+valSize-1]
    valY = Y[trainSize:trainSize+valSize-1]
    testX = X[valSize:valSize+testSize-1]
    testY = Y[valSize:valSize+testSize-1]

    # sweep space to find optimal parameter gamma
    gamma = sweepOptimalVal(trainX, valX, trainY, valY, svmType, param = 'gamma', \
            minVal = -3, maxVal = 3, steps = 2, stepType = 'log')
    #gamma = findOptimalVal(trainX, valX, trainY, valY, param = 'gamma')


# Find N clusters in X data
def cluster(X,N):

    kcluster = KMeans(n_clusters=N, init='k-means++', n_init=10,\
            max_iter=300, tol=0.0001, precompute_distances=True, verbose=0,\
            random_state=None, copy_x=True, n_jobs=1)

    kcluster.fit(X)
    
    return kcluster


            
# sweeps through possible values for a parameter and plots the accuracy
# for valdiation and training for values tested
def sweepOptimalVal(trainX, valX, trainY, valY, svmType, param = 'lambda', \
        minVal = -6, maxVal = 3, steps = 100, stepType = 'log'):
    
    # check if logarithmic or linear spacing
    if stepType == 'log':
        testVals = np.logspace(minVal,maxVal,steps)
    elif stepType == 'lin':
        testVals = np.linspace(minVal,maxVal,steps)
    else:
        print "Error: unknown stepType for sweepOptimalVal function call"
        print "Please slect 'log' or 'lin'"
        exit()

    # create vectors to record accuracies
    trainAccuracy = np.zeros(steps)
    valAccuracy = np.zeros(steps)

    # initialize classifier
    clf = []

    # FIXME: Now only testing gamma
    for i,val in enumerate(testVals):

        print "\nTraining with gamma = " + str(val)

        # check SVM type
        if svmType == 'one-vs-the-rest':
            clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True,\
                fit_intercept=True, intercept_scaling=1, loss='l2', \
                multi_class='ovr', penalty='l2', random_state=None, \
                    tol=0.0001, verbose=0)
            clf.fit(trainX, trainY)

        elif svmType == 'one-against-one':
            clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, \
                    degree=3, gamma=val, kernel='rbf', max_iter=-1, \
                    probability=False, random_state=None, shrinking=True, \
                    tol=0.001, verbose=False)
            clf = svm.SVC(gamma = val)
            clf.fit(trainX, trainY)

        else:
            print "Error: unrecognized svmType"
            print "Please select 'one-vs-the-rest' or 'one-against-one'"
            exit()
        
        # record training and validation accuracy
        trainAccuracy[i] = \
               getClassificationAccuracy(clf.predict(trainX), trainY)
        print "Training Accuracy = " + str(trainAccuracy[i])
        valAccuracy[i] = \
               getClassificationAccuracy(clf.predict(valX), valY)
        print "Validaiton Accuracy = " + str(valAccuracy[i])

    # plot training results
    if stepType == 'log':
        plt.semilogx(testVals, trainAccuracy)
    else:
        plt.plot(testVals, trainAccuracy)

    plt.xlabel(param)
    plt.ylabel("Training Accuracy")
    plt.show()
   
    # plot validation results
    if stepType == 'log':
        plt.semilogx(testVals, valAccuracy)
    else:
        plt.plot(testVals, valAccuracy)
    
    plt.xlabel(param)
    plt.ylabel("Validation Accuracy")
    plt.show()

    return testVals[np.argmin(valAccuracy)]


# determines the classification accuracy of Yprime with actual data Y
def getClassificationAccuracy(Yprime, Y):
    N = len(Y)
    success = 0
    for i in xrange(N):
        if Yprime[i] == Y[i]:
            success += 1
    return float(success) / float(N)

# function to train SVM data
def trainSVM(X, Y, svmType = 'one-vs-the-rest', kernel = 'rbf', gamma = 1.0,\
        C = 1.0):

    # initialize clf
    clf = None

    # check SVM type
    if svmType == 'one-vs-the-rest':
        clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True,\
            fit_intercept=True, intercept_scaling=1, loss='l2', \
            multi_class='ovr', penalty='l2', random_state=None, \
            tol=0.0001, verbose=0)
        clf.fit(X, Y)

    elif svmType == 'one-against-one':
        clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, \
                degree=3, gamma=100.0, kernel='rbf', max_iter=-1, \
                probability=False, random_state=None, shrinking=True, \
                tol=0.001, verbose=False)
        clf.fit(X, Y)
   


if __name__ == "__main__":
    
    # load user features for clustering
    Xdata = []
    filename = '../data/user_features_1.csv'

    f = open(filename,'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        featureVector = []
        words = line.split()
        for word in words[1:]:
            featureVector.append( float(word) )
        Xdata.append(featureVector)

    # run clustering algorithm on X data
    numclusters = 6
    model = cluster(Xdata, numclusters)

    classes = model.predict(Xdata)

    output = np.zeros(numclusters)

    for pt in classes:
        output[pt] += 1

    print "Cluster results:"
    print output
