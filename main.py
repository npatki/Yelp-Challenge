from collections import defaultdict
from sklearn import (
    cluster, linear_model, ensemble, neighbors, preprocessing, mixture
)
from util import *
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

"""This is the main script that does the end-to-end testing.

1. Take the user feature vector partitions and designate 1
   as the testing, 1 as the validation, and 2 for training.
2. Cluster the user feature vectorss.
3. For each cluster:
   3.1 Calculate the new business ratings for users only
       belonging to that cluster.
   3.2 Break the business vectors into 4 partitions: 1 for testing,
       1 for validation, 2 for training.
   3.2 Run the regression for that cluster on the training data. Use
       validation data to select hyperparameters of the model.
   3.3 Report the overall test results.
4. Use the validation set to run the previous steps multiple times.
   Choose the desired # of clusters (hyperparameter).
5. Report the overall test error."""


def users_validation(predictor, maximum=float('inf')):
    """Perform end-to-end analysis on the users validation set.
    
    :param predictor: a function that takes in a user vector and
                      user ID, and outputs the guessed rating as
                      well as the actual rating for each restaurant
                      the user has reviewed.
    :param maximum: an optional int used to save time. Overall
                    analysis is only done on this many users.
    :returns a float representing the average error made in rating
             per user"""
    test_vectors, ID = get_user_vectors('validate')
    total_users = min(len(ID), maximum)
    total_error = 0.0

    ct = 0
    for i, v in enumerate(ID):
        ct += 1
        if ct == maximum:
            break
        if ct%100 == 0:
            print 'done %d users' % ct

        guess, actual = predictor(test_vectors[i], v)
        total_error += get_error(actual, guess)

    return total_error/float(total_users)

def gaussianMixture(num_clusters, learner):
 
    """Fits a gaussian mixture model for users, taking the maximum
    likelihood estimate to determine the cluster before learning 
    for different groups.

    :param num_clusters: an int representing the number of gaussians
    :param learner: a function that takes in a set of user IDs and
                    outputs a prediction function that's capable
                    of inferring a user's rating.
                    See methods below.
    :returns a function that can take in a user vector and user id,
             and outputs both the guessed rating and the actual rating
             for each restaurant the user has reviewed."""

    vectors, ID = get_user_vectors('train')
    GMM = mixture.GMM(n_components=num_clusters)
    
    # scale features to mean of 0 and standard devaition 1
    scaler = preprocessing.StandardScaler().fit(vectors)
    scaled_vectors = scaler.transform(vectors)
    
    GMM.fit(scaled_vectors)

    classes = GMM.predict(scaled_vectors)

    # create a dictionary where a cluster # maps to a list of
    # user IDs belonging to that cluster
    groups = defaultdict(list)
    
    for i, v in enumerate(classes):
        groups[v].append(ID[i])

    # these are the corresponding functions to call for each group
    predictors = [0]*num_clusters

    for g in groups:
        user_set = set(groups[g])
        predictors[g] = learner(user_set)
    
    # this is a function that does the end-to-end prediction:
    # first find the cluster the user belongs to
    # then find all the restauratns this user has gone to and
    # see how the predictor's guesses compare
    def fn(user_vector, user_id):
        scaled_vector = scaler.transform(user_vector)
        c = GMM.predict([scaled_vector])[0]
        ratings, NA = compute_ratings(set([user_id]))
        
        inp, actual, NA = get_biz_vectors('all', ratings)
        guess = predictors[c](inp)
        return guess, actual

    return fn

def user_history(a, b):
    def fn(user_vector, user_id):
        ratings, NA = compute_ratings(set([user_id]))
        _, actual, _ = get_biz_vectors('all', ratings)
        output = sum(ratings.values())/float(len(ratings))
        output = np.array([output])
        return output, actual
    return fn

def bayesianGaussianMixture(num_clusters, learner):
 
    """Fits a gaussian mixture model for users, taking the maximum
    likelihood estimate to determine the cluster before learning 
    for different groups.

    :param num_clusters: an int representing the number of gaussians
    :param learner: a function that takes in a set of user IDs and
                    outputs a prediction function that's capable
                    of inferring a user's rating.
                    See methods below.
    :returns a function that can take in a user vector and user id,
             and outputs both the guessed rating and the actual rating
             for each restaurant the user has reviewed."""

    vectors, ID = get_user_vectors('train')
    GMM = mixture.GMM(n_components=num_clusters)
    
    # scale features to mean of 0 and standard devaition 1
    scaler = preprocessing.StandardScaler().fit(vectors)
    scaled_vectors = scaler.transform(vectors)
    
    GMM.fit(scaled_vectors)

    classes = GMM.predict_proba(scaled_vectors)

    # create a dictionary where a cluster # maps to a list of
    # user IDs belonging to that cluster
    user_weights = []
    groups = []
    for j in xrange(num_clusters):
        weight_vector = []
        groups.append([])
        for i in xrange( len(scaled_vectors) ):
            class_component = round(classes[i][j],6)
            if class_component != 0:
                groups[j].append(ID[i])
                weight_vector.append( class_component )
        
        # normalize weights
        weight_vector = np.array(weight_vector)
        weight_vector /= sum(weight_vector)
        
        # input final user weights into dictionary
        user_weights.append(dict())
        for k, weight in enumerate(weight_vector):
            user = groups[j][k]
            user_weights[j][user] = weight

    # these are the corresponding functions to call for each group
    predictors = [0]*num_clusters

    # train predictors - each predictor uses almost the entire data set with
    # weights corresponding associated with each user
    for g, weights in enumerate(user_weights):
        user_set = set(groups[g])
        predictors[g] = learner(user_set, weights)
    
    # this is a function that does the end-to-end prediction:
    # first find the cluster the user belongs to
    # then find all the restauratns this user has gone to and
    # see how the predictor's guesses compare
    def fn(user_vector, user_id):
        scaled_vector = scaler.transform(user_vector)
        new_weights = GMM.predict_proba([scaled_vector])[0]
        ratings, NA = compute_ratings(set([user_id]))

        inp, actual, NA = get_biz_vectors('all', ratings)
        
        # form bayesian guess
        guess = 0
        for g, predictor in enumerate(predictors):
            guess += new_weights[g] * predictor(inp)
       
        return guess, actual

    return fn

def kNeighbors(n_neighbors, learner):
    """Use the k nearest-neighbors of the input vector to learn
    the user's likes/dislikes.

    :param n_neighbors: an int representing the number of nearest
                        neighbors to look at.
    :param learner: a function that takes in a set of user IDs and
                    output a prediction function that's capable of
                    inferring a user's rating.
    :returns a function that can take in a user vector and user id,
             and output both the guessed rating and the actual rating
             for each restaurant the user has reiewed"""

    vectors, ID = get_user_vectors('train')
    kNeighbors = neighbors.NearestNeighbors(n_neighbors)
    # calling fit means these vectors will be returned with a query
    # is made for the k nearest neighbors
    kNeighbors.fit(vectors)

    def fn(user_vector, user_id):
        n_index = kNeighbors.kneighbors(user_vector, n_neighbors,
            return_distance=False)
        user_set = set()

        # n_index[0] contains indicies which correspond to the n
        # nearest users from the training set to user_vector
        for i in n_index[0]:
            user_set.add(ID[i])

        # user the nearest neighbors to perform the learning
        predictor = learner(user_set)
        ratings, NA = compute_ratings(set([user_id]))

        inp, actual, NA = get_biz_vectors('all', ratings)
        guess = predictor(inp)

        return guess, actual

    return fn

def kMeans(num_clusters, learner):
    """Perform K-Means clustering to sort the users before
    learning for different groups.

    :param num_clusters: an int representing the number of clusters
    :param learner: a function that takes in a set of user IDs and
                    outputs a prediction function that's capable
                    of inferring a user's rating.
                    See methods below.
    :returns a function that can take in a user vector and user id,
             and outputs both the guessed rating and the actual rating
             for each restaurant the user has reviewed."""
    vectors, ID = get_user_vectors('train')
    kMeans = cluster.KMeans(n_clusters=num_clusters)
    
    # scale features to mean of 0 and standard devaition 1
    scaler = preprocessing.StandardScaler().fit(vectors)
    scaled_vectors = scaler.transform(vectors)
    
    kMeans.fit(scaled_vectors)

    classes = kMeans.predict(scaled_vectors)

    # create a dictionary where a cluster # maps to a list of
    # user IDs belonging to that cluster
    groups = defaultdict(list)
    
    for i, v in enumerate(classes):
        groups[v].append(ID[i])

    # these are the corresponding functions to call for each group
    predictors = [0]*num_clusters

    for g in groups:
        user_set = set(groups[g])
        predictors[g] = learner(user_set)
    
    # this is a function that does the end-to-end prediction:
    # first find the cluster the user belongs to
    # then find all the restauratns this user has gone to and
    # see how the predictor's guesses compare
    def fn(user_vector, user_id):
        scaled_vector = scaler.transform(user_vector)
        c = kMeans.predict(scaled_vector)[0]
        ratings, NA = compute_ratings(set([user_id]))

        inp, actual, NA = get_biz_vectors('all', ratings)
        guess = predictors[c](inp)
        return guess, actual

    return fn

######### These are the learning algorithms for regression #########
# All of these take in a set of user IDs, perform the learning algo,
# and return a function that's able to predict rating based on the
# business feature vector

def mle(user_set, weights = None):
    """Maximum liklihood estimate learner to use as a baseline."""
    
    ratings, biz_weights = compute_ratings(user_set, weights)
    
    if biz_weights == None:
        avg = sum(ratings.values())/float(len(ratings))
    else:
        # form weighted average
        tally = 0
        weight = 0
        for key, value in ratings.items():
            tally += biz_weights[key] * value
            weight += biz_weights[key]
        avg = tally / weight

    def predict(vals):
        out = [avg]*len(vals)
        return np.array(out)

    return predict

def random_forests(user_set, weights = None):
    """ Random forest learner."""
    # TODO set hyperparameters using validation set
    l = ensemble.RandomForestRegressor()
    ratings, biz_weights = compute_ratings(user_set, weights)
    X_train, Y_train, W_train = get_biz_vectors('train', ratings, biz_weights)
    l.fit(X_train, Y_train, W_train)
    return l.predict


def ridge(user_set, weights = None):
    """Ridge regression learner that uses the validation set to tune alpha."""
    alphas = np.arange(0.1, 1, 0.1)

    ratings, biz_weights = compute_ratings(user_set, weights)
    X_train, Y_train, W_train = get_biz_vectors('train', ratings, biz_weights)
    X_validate, Y_validate, W_validate = \
            get_biz_vectors('validate', ratings, biz_weights)

    best_val = float('inf')
    best_predictor = None

    for alpha in alphas:
        l = linear_model.Ridge(alpha=alpha, normalize=False)
        l.fit(X_train, Y_train, W_train)

        Y_predicted = l.predict(X_validate)
        val_error = get_error(Y_validate, Y_predicted, W_validate)

        if val_error < best_val:
            best_val = val_error
            best_predictor = l.predict 

    return best_predictor

def lasso(user_set):
    """Lasso learner that uses the validation set to tune alpha."""
    alphas = np.arange(0.1, 1, 0.1)

    ratings, NA = compute_ratings(user_set)
    X_train, Y_train, NA = get_biz_vectors('train', ratings)
    X_validate, Y_validate, NA = get_biz_vectors('validate', ratings)

    best_val = float('inf')
    best_predictor = None

    for alpha in alphas:
        l = linear_model.Lasso(alpha=alpha, normalize=False)
        l.fit(X_train, Y_train)

        Y_predicted = l.predict(X_validate)
        val_error = get_error(Y_validate, Y_predicted)

        if val_error < best_val:
            best_val = val_error
            best_predictor = l.predict 

    return best_predictor

def bayesian_ridge(user_set):
    # TODO set hyperparameters using validation set
    ratings, NA = compute_ratings(user_set)
    X_train, Y_train, NA = get_biz_vectors('train', ratings)

    l = linear_model.BayesianRidge()
    l.fit(X_train, Y_train)
    return l.predict

def run(cluster_method, regression_method, hyperparam):
    predictor = cluster_method(hyperparam, regression_method)

    print 'doing prediction'
    t0 = time.clock()
    print users_validation(predictor)
    t1 = time.clock()
    print 'time elapsed %f' % (t1 - t0)


if __name__ == '__main__':
    # uncomment one of these to analyze
    # predictor = kNeighbors(200, bayesian_ridge)
    # predictor = kMeans(2, lasso)
    # predictor = kMeans(1, mle)
    # predictor = kMeans(2, bayesian_ridge)

    # NOTE: Baysian gaussian can only be used with mle and ridge
    # predictor = bayesianGaussianMixture(2, ridge)

    run(kMeans, mle, 2)
    #predictor = bayesianGaussianMixture(2, ridge)

    predictor = bayesianGaussianMixture(cluster_num, )
    print users_validation(predictor, maximum=20)
