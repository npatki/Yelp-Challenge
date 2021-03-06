#!/usr/local/bin/python
import parser
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GMM

"""
Determines user clusters
"""
def determineClusters(filename, num_clusters = 6, cluster_type = 'kmeans', 
        analyze_clusters = False):

    # initialize vectors
    user_IDs = []
    feature_vectors = []

    # load user data from CSV file
    with open(filename, 'rb') as csvfile:

        # read all lines in the file
        lines = csvfile.readlines()

        # read the header
        header = lines[0].split()

        # save user IDs, load feature vectors
        for line in lines[1:]:

            # load in line elements as strings
            elements = line.split()
            user_IDs.append( elements[0] )

            # convert feature vector arguments to floats and add to vector
            feature_vector = np.zeros( len(elements) - 1)
            for (i, element) in enumerate(elements[1:]):
                feature_vector[i] += float(element)

            # add feature vector to list
            feature_vectors.append(feature_vector)

    # first scale data for clustering
    scaler = preprocessing.StandardScaler().fit(feature_vectors)
    scaled_features = scaler.transform(feature_vectors)

    # train cluster generator as defined by cluster_type
    if cluster_type is 'kmeans':

        # define generator using k-means clustering
        cluster_gen = KMeans(n_clusters=num_clusters, init='k-means++', 
                n_init=10, max_iter=300, tol=0.0001, 
                precompute_distances=True, verbose=0, random_state=None, 
                copy_x=True, n_jobs=1)

    elif cluster_type is 'GMM':

        # define generator using mixture of gaussians (GMM)
        cluster_gen = GMM(n_components=num_clusters, covariance_type='diag', 
                random_state=None, thresh=0.01, min_covar=0.001, n_iter=100, 
                n_init=1, params='wmc', init_params='wmc')

    # train cluster generator
    cluster_gen.fit(scaled_features)
    clusters = cluster_gen.predict(scaled_features)

    # output cluster information if requested
    if analyze_clusters:
        printClusterStats(clusters, num_clusters, feature_vectors, header)

    # form dictionary of (user_id, cluster)
    cluster_dictionary = dict( zip(user_IDs, clusters) )

    return cluster_dictionary

"""
Prints cluster information
"""
def printClusterStats(clusters, num_clusters, feature_vectors, header):

    # determine the number of users in each cluster
    occurances = np.zeros(num_clusters)
    for cluster in clusters:
        occurances[cluster] += 1
    
    # display number of users in each cluster
    for i in xrange(num_clusters):
        print "Number of users in cluster " + str(i) + " = " + \
                str(occurances[i])
    
    # create arrays for characteristic feature vectors
    rep_feature_vectors = []
    feature_vector_length = len(feature_vectors[0])
    
    for i in xrange(num_clusters):
        rep_feature_vectors.append( np.zeros(feature_vector_length) )

    # determine the characteristic (average) feature vector for each cluter
    for i in xrange(len(feature_vectors)):
        
        # load relevant information
        cluster = clusters[i]
        feature_vector = feature_vectors[i]

        # add feature vector attributes to cluster
        rep_feature_vectors[cluster] += feature_vector / occurances[cluster]

    # normalize feature vectors by number of elements and print to screen
    for i in xrange(num_clusters):
        
        #rep_feature_vectors[i] /= occurances[i]
        
        print "_______________________________________________"
        print "Characteristic features for cluster " + str(i) + ":"
        
        rep_feature_vector = rep_feature_vectors[i]
        for feature_name, feature_val in zip(header[1:], rep_feature_vector):
            print feature_name + " = " + str(feature_val)

if __name__ == "__main__":
    
    # run clustering algorithm on for first file
    user_clusters = determineClusters( "../data/user_features_0.csv", 6, 
            cluster_type='GMM', analyze_clusters = True)

