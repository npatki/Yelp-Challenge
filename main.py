from cluster_solver import (
    compute_rankings, get_input_output, ensemble
)
from collections import defaultdict
from sklearn import cluster, linear_model
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

mapping = {
    'test': [0],
    'validate': [1],
    'train': [2, 3],
    'all': [0, 1, 2, 3]
}

def get_biz_vectors(process, ratings):
    partition_list = mapping[process]
    X_out = []
    Y_out = []

    for i in partition_list:
        filename = '../data/biz_features_%d.csv' % i
        X, Y = get_input_output(ratings, filename)
        X_out.extend(X)
        Y_out.extend(Y)

    return X_out, Y_out

def get_user_vectors(process):
    partition_list = mapping[process]
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

                vector = [float(j) for j in elements[15:]]
                X_out.append(vector)
    
    return X_out, ID_out

def get_error(Y_actual, Y_predicted):
    Y_actual = np.array(Y_actual)

    Y_predicted[Y_predicted > 5.0] = 5.0
    Y_predicted[Y_predicted < 1.0] = 1.0

    return sum(abs(Y_predicted-Y_actual))/len(Y_actual)

def best_clusterize(num_clusters, simple=False):
    vectors, ID = get_user_vectors('train')
    kMeans = cluster.KMeans(n_clusters=num_clusters)

    kMeans.fit(vectors)

    classes = kMeans.predict(vectors)
    groups = defaultdict(list)

    for i, v in enumerate(classes):
        groups[v].append(ID[i])

    predictors = [0]*num_clusters

    for g in groups:
        user_set = set(groups[g])

        if simple:
            predictors[g] = simple_predictor(user_set)
        else:
            # val, test, clf = get_best_alpha(user_set)
            # clf = linear_model.BayesianRidge()
            clf = ensemble.RandomForestRegressor()
            rankings = compute_rankings(user_set)
            X_train, Y_train = get_biz_vectors('train', rankings)
            clf.fit(X_train, Y_train)
            predictors[g] = clf.predict
            # print val, test

    test_vectors, ID = get_user_vectors('validate')

    total_users = len(ID)
    total_error = 0.0

    ct = 0

    for i, v in enumerate(ID):
        ct += 1

        if ct == 250:
            break

        ratings = compute_rankings(set([v]))
        user_vector = test_vectors[i]
        c = kMeans.predict(user_vector)[0]
        
        inp, actual = get_biz_vectors('all', ratings) 
        guess = predictors[c](inp)
        total_error += get_error(actual, guess)

    print total_error/250
        
    # get biz_vectors for just the user
    # predict for just the user

def get_best_alpha(user_set):
    alphas = np.arange(0.1, 1, 0.1)

    best_val = float('inf')
    best_clf = None
    best_test = None

    for alpha in alphas:
        val, test, clf = lasso(user_set, alpha)
        if val < best_val:
            best_val = val
            best_clf = clf
            best_test = test

    return best_val, best_test, best_clf

def simple_predictor(user_set):
    """Simple predictor as a baseline: just guess the average of everything in
    the set."""
    rankings = compute_rankings(user_set)
    
    avg = sum(rankings.values())/float(len(rankings))

    def predict(vals):
        out = [avg]*len(vals)
        return np.array(out)

    return predict

def lasso(user_set, alpha):
    rankings = compute_rankings(user_set)

    X_train, Y_train = get_biz_vectors('train', rankings)
    X_validate, Y_validate = get_biz_vectors('validate', rankings)
    X_test, Y_test = get_biz_vectors('test', rankings)

    clf = linear_model.Lasso(alpha=alpha, normalize=False)
    clf.fit(X_train, Y_train)
    
    Y_predicted_train = clf.predict(X_train)
    Y_predicted_validate = clf.predict(X_validate)
    Y_predicted_test = clf.predict(X_test)

    train_error = get_error(Y_train, Y_predicted_train)
    validate_error = get_error(Y_validate, Y_predicted_validate)
    test_error = get_error(Y_test, Y_predicted_test)
    
    return (validate_error, test_error, clf)


if __name__ == '__main__':
    best_clusterize(10, simple=False)
