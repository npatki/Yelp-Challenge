from collections import defaultdict
from sklearn import cluster, linear_model, ensemble
from util import *
import numpy as np

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

        guess, actual = predictor(test_vectors[i], v)
        total_error += get_error(actual, guess)

    return total_error/float(total_users)


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
    kMeans.fit(vectors)

    classes = kMeans.predict(vectors)

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
        c = kMeans.predict(user_vector)[0]
        ratings = compute_ratings(set([user_id]))

        inp, actual = get_biz_vectors('all', ratings)
        guess = predictors[c](inp)
        return guess, actual

    return fn

######### These are the learning algorithms for regression #########
# All of these take in a set of user IDs, perform the learning algo,
# and return a function that's able to predict rating based on the
# business feature vector

def mle(user_set):
    """Maximum liklihood estimate learner to use as a baseline."""
    ratings = compute_ratings(user_set)
    avg = sum(ratings.values())/float(len(ratings))

    def predict(vals):
        out = [avg]*len(vals)
        return np.array(out)

    return predict

def random_forests(user_set):
    """Random forest learner."""
    # TODO set hyperparameters using validation set
    l = ensemble.RandomForestRegressor()
    ratings = compute_ratings(user_set)
    X_train, Y_train = get_biz_vectors('train', ratings)
    l.fit(X_train, Y_train)
    return l.predict

def lasso(user_set):
    """Lasso learner that uses the validation set to tune alpha."""
    alphas = np.arange(0.1, 1, 0.1)

    ratings = compute_ratings(user_set)
    X_train, Y_train = get_biz_vectors('train', ratings)
    X_validate, Y_validate = get_biz_vectors('validate', ratings)

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

    return l.predict

def bayesian_ridge(user_set):
    # TODO set hyperparameters using validation set
    ratings = compute_ratings(user_set)
    X_train, Y_train = get_biz_vectors('train', ratings)

    l = linear_model.BayesianRidge()
    l.fit(X_train, Y_train)
    return l.predict


if __name__ == '__main__':
    # uncomment one of these to analyze
    predictor = kMeans(2, random_forests)
    # predictor = kMeans(2, lasso)
    # predictor = kMeans(2, mle)
    # predictor = kMeans(2, bayesian_ridge)

    print users_validation(predictor, maximum=250)
