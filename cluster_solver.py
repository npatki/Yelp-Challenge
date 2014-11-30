from collections import defaultdict
from sklearn import linear_model, ensemble
from user_extractor import LoadReviewInformation

import numpy as np


REVIEW_FILE = '../data/yelp_academic_dataset_review.json'
CLUSTER_BASE = '../data/cluster_'

reviews = LoadReviewInformation(REVIEW_FILE)


def rerate_businesses(cluster_number):
    """Given a cluster of users, we need to recalculate the
    rating for the business using only these users. This 
    method recalculates and returns the results.
    
    :param cluster_number: an int describing the cluster number.
           The number is put in place of x in
           ../data/cluster_x.txt
    
    :returns: a dictionary mapping business_id --> avg rating
            for only the users in the given cluster"""

    # create the set of users we'll be looking at
    cluster_file = CLUSTER_BASE + str(cluster_number) + '.txt'

    user_set = set()

    with open(cluster_file, 'r') as f:
        for line in f:
            user_set.add(line.strip())

    return compute_rankings(user_set)

def compute_rankings(user_set):
    """Method that recalculates rankings of a business to reflect
    only the user IDs that are in user_set.
    
    :param user_set: a set containing strings representing user IDs

    :returns a dictionary mapping biz_id --> avg rating for only
        the users in user_set"""
    # intermediate dictionary that maps biz_id --> list of ratings
    # given to it by members of user_set
    ratings = defaultdict(list)

    for user in reviews:
        if user in user_set:
            for biz, rating in reviews[user]:
                ratings[biz].append(rating)
    
    # the output should map biz_id --> average of the list
    new_ratings = {}
    for biz, values in ratings.items():
        new_ratings[biz] = sum(values)/float(len(values))

    return new_ratings

def get_input_output(ratings, filename):
    """Grab and format the feature vectors and outputs so that
    the data is ready to be fed into regression.
    
    :param ratings: the output of rerate_businesses, the
        mapping business_id --> recalculated rating
    :param filename: the filename containing the business feature
        vectors"""
    X = []
    Y = []

    with open(filename, 'r') as f:
        for line in f:
            info = line.strip().split()

            # info[0] contains the ID of the business
            # make sure it's in the ratings for this cluster
            if info[0] in ratings:
                # info[-1] is the overall rating
                # we don't need it because we're trying to predict
                # the cluster's rating
                values = [float(j) for j in info[1:-1]]
                X.append(values)
                Y.append(ratings[info[0]])

    return X, Y


# TODO: this does training/testing. move to train-validate-test scheme
def regress(ratings, test=0):
    """Use the recalulated ratings to form the new feature vectors
    and perform the regression.
    
    :param ratings: the output of rerate_businesses, the 
        mapping business_id --> recalculated rating
    :param test: the fold to use for testing
        the other partitions are currently used to train
    
    :returns: the regression object after it has been trained
        on the data"""
    X_train = []
    Y_train = []

    # create the inputs and outputs by going through all training data
    for i in range(4):
        if i == test:
            continue

        filename = '../data/biz_features_%d.csv' % i
        X, Y = get_input_output(ratings, filename)
        X_train.extend(X)
        Y_train.extend(Y)

    # create and train the linear model
    # TODO: try other forms of regresssion
    clf = linear_model.LinearRegression(normalize=True)
    clf.fit(X_train, Y_train)
    
    # cap the predictions to values between 1 and 5
    Y_prime = clf.predict(X_train)
    Y_prime[Y_prime > 5.0] = 5.0
    Y_prime[Y_prime < 1.0] = 1.0

    Y_train = np.array(Y_train)

    print 'Training error per review:'
    print sum(abs(Y_prime-Y_train))/len(Y_train)

    testfile = '../data/biz_features_%d.csv' % test
    X_test, Y_test = get_input_output(ratings, testfile)

    # cap the predictions to values between 1 and 5
    Y_prime = clf.predict(X_test)
    Y_prime[Y_prime > 5.0] = 5.0
    Y_prime[Y_prime < 1.0] = 1.0

    Y_test = np.array(Y_test)

    print 'Testing error per review:'
    print sum(abs(Y_prime-Y_test))/len(Y_test)

    return clf

if __name__ == '__main__':
    rerate = rerate_businesses(0)
    regress(rerate, test=0)
