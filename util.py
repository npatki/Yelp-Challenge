from collections import defaultdict
from user_extractor import LoadReviewInformation
import numpy as np

REVIEW_FILE = '../data/yelp_academic_dataset_review.json'
reviews = LoadReviewInformation(REVIEW_FILE)

MAPPING = {
    'test': [0],
    'validate': [1],
    'train': [2, 3],
    'all': [0, 1, 2, 3]
}

def get_error(Y_actual, Y_predicted, weights = None):
    """Calculates the error in prediction.
    
    :param Y_actual: an array of expected outputs
    :param Y_predicted: an array of predicted outputs
    :returns a float representing the average error made per review"""
    Y_actual = np.array(Y_actual)

    # overshooting means the prediction is the highest rating: 5.0
    Y_predicted[Y_predicted > 5.0] = 5.0
    # undershooting means the prediction is the lowest rating: 1.0
    Y_predicted[Y_predicted < 1.0] = 1.0

    if weights == None:
        return sum(abs(Y_predicted-Y_actual))/len(Y_actual)
    else:
        return sum(abs(Y_predicted-Y_actual)*weights)

def compute_ratings(user_set, weights = None):
    """Method that recalculates the ratings of a business to 
    reflect only the users in the user_set
    
    :param user_set: a set containing strings representing user IDs
    :returns a dictionary mapping biz_id --> avg rating as determined
             only by users in user_set"""
    
    # intermediate dictionary that maps biz_id --> list of ratings
    # given to it by members of user_set
    ratings = defaultdict(list)

    for user in user_set:
        for biz, rating in reviews[user]:
            ratings[biz].append(rating)
    
    # the output should map biz_id --> average of the list
    new_ratings = {}
    new_weights = {}
    if weights == None:
        new_weights = None
        for biz, values in ratings.items():
            new_ratings[biz] = sum(values)/float(len(values))
    else:
        # create weighted ratings
        for biz, values in ratings.items():
            new_ratings[biz] = 0
            new_weights[biz] = 0
            for i, value in enumerate(values):
                new_ratings[biz] += weights[i] * value
                new_weights[biz] += weights[i]
            
            new_ratings[biz] /= new_weights[biz]
    
    return new_ratings, new_weights

def get_user_vectors(process):
    """Get vectors representing the users.
    
    :param process: either 'test', 'validate', 'train', or 'all'
    :returns X_out, ID_out, a list of user vectors for that process,
            followed by a list of corresponding user IDs"""
    partition_list = MAPPING[process]
    X_out = []
    ID_out = []

    for i in partition_list:
        filename = '../data/user_features_%d.csv' % i

        with open(filename, 'rb') as csvfile:
            lines = csvfile.readlines()
            header = lines[0].split()

            for line in lines[1:]:
                elements = line.split()
                ID_out.append(elements[0].strip())

                vector = [float(j) for j in elements[1:]]
                X_out.append(vector)
    
    return X_out, ID_out

def get_biz_vectors(process, ratings, weights = None):
    """Get the vectors representing businesses.
    
    :param process: either 'test', 'validate', 'train' or 'all'
    :param ratings: a map of biz_id --> ratings
    :returns X_out, Y_out, a list of input feature vectors followed
            by a list of expected output"""
    partition_list = MAPPING[process]
    X_out = []
    Y_out = []
    biz_out = []

    for i in partition_list:
        filename = '../data/biz_features_%d.csv' % i
        X, Y, biz_list = get_biz_input_output(ratings, filename)
        X_out.extend(X)
        Y_out.extend(Y)
        biz_out.extend(biz_list)

    if weights is None:
        W_out = None
    else:
        W_out = []
        for biz in biz_out:
            W_out.append( weights[biz] )
        W_out = np.array(W_out)

    return X_out, Y_out, W_out

def get_biz_input_output(ratings, filename):
    """Grab and format the business feature vectors and outputs so that
    data is ready to be fed into regression.
    
    :param ratings: the output of compute_ratings, a map
            from biz_id --> recalculated average rating
    :param filename: the filename containing the business feature 
            vectors
    :returnx X, Y a list of input feature vectors follow by a list
            of expected output"""

    X = []
    Y = []
    biz_list = []

    with open(filename, 'r') as f:
        for line in f:
            info = line.strip().split()

            # info[0] contains the ID of the business
            biz = info[0]
            if biz in ratings:
                # info[-1] is the overall rating of the biz
                # we want to predict the recomputed rating
                values = [float(j) for j in info[1:-1]]
                X.append(values)
                Y.append(ratings[biz])
                biz_list.append(biz)

    return X, Y, biz_list
