from collections import defaultdict
from user_extractor import LoadReviewInformation, LoadBusinessFeatures
import numpy as np
from copy import deepcopy as copy
# from pandas.tools.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import matplotlib as mpl

REVIEW_FILE = '../data/yelp_academic_dataset_review.json'
reviews = LoadReviewInformation(REVIEW_FILE)

MAPPING = {
    'test': [0],
    'validate': [1],
    'train': [2, 3],
    'all': [0, 1, 2, 3]
}

BIZ_DICT = {}
for name in MAPPING.keys():
   
    BIZ_DICT[name] = {}
    for m in MAPPING[name]:
        filename = '../data/biz_features_%d.csv' % m
        BIZ_DICT[name].update( LoadBusinessFeatures(filename) )

def get_error(Y_actual, Y_predicted):
    """Calculates the error in prediction.
    
    :param Y_actual: an array of expected outputs
    :param Y_predicted: an array of predicted outputs
    :returns a float representing the average error made per review"""
    Y_actual = np.array(Y_actual)

    # overshooting means the prediction is the highest rating: 5.0
    Y_predicted[Y_predicted > 5.0] = 5.0
    # undershooting means the prediction is the lowest rating: 1.0
    Y_predicted[Y_predicted < 1.0] = 1.0

    return sum(abs(Y_predicted-Y_actual))/len(Y_actual)

def compute_ratings(user_set):
    """Method that recalculates the ratings of a business to 
    reflect only the users in the user_set
    
    :param user_set: a set containing strings representing user IDs
    :returns a dictionary mapping biz_id --> avg rating as determined
         only by users in user_set"""
    
    # intermediate dictionary that maps biz_id --> list of ratings
    # given to it by members of user_set
    ratings = defaultdict(list)

    for user in user_set:
        if user in reviews:
            for biz, rating in reviews[user]:
                ratings[biz].append(rating)
    
    # the output should map biz_id --> average of the list
    new_ratings = {}
    for biz, values in ratings.items():
        new_ratings[biz] = sum(values)/float(len(values))
    
    return new_ratings

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

def get_biz_vectors(process, ratings):
    """Get the vectors representing businesses.
    
    :param process: either 'test', 'validate', 'train' or 'all'
    :param ratings: a map of biz_id --> ratings
    :returns X_out, Y_out, a list of input feature vectors followed
            by a list of expected output"""
    dictionary = BIZ_DICT[process]
    
    X_out = []
    Y_out = []

    for biz in ratings:
        if biz in dictionary:
            X_out.append(dictionary[biz])
            Y_out.append(ratings[biz])

    return X_out, Y_out



def plot_clusters(Xdata, labels, process, ax=None, color=None,
                     use_columns=False, xticks=None, colormap=None,
                     **kwds):
    """Plot clusters representing the users.
    
    :param Xdata: either 'test', 'validate', 'train', or 'all'
    :returns: the handle of the clustering plot"""
    partition_list = MAPPING[process]

    # determine cluster parameters
    n = len(labels)
    class_min = float(min(labels))
    class_max = float(max(labels))

    # Read header
    filename = '../data/user_features_%d.csv' % partition_list[0]
    with open(filename, 'rb') as csvfile:
        lines = csvfile.readlines()
        header = lines[0].split()
    header = header[1:]

    # set the x axis to equally spaced
    x = range(len(Xdata[0]))

    # set plotting parameters
    Colorm = plt.get_cmap(colormap)
    fig = plt.figure()
    ax = plt.gca()

    # plot data points
    for i in range(n):
        y = Xdata[i]
        kls = labels[i]
        ax.plot(x, y, color=Colorm((kls - class_min)/(class_max-class_min)), **kwds)

    # plot vertical lines
    for i in x:
        ax.axvline(i, linewidth=1, color='black')

    # format plot
    ax.set_xticks(x)
    #ax.set_xticklabels(header, rotation = 'vertical')
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc='upper right')
    ax.grid()

    bounds = np.linspace(class_min,class_max,10)
    #cax,_ = mpl.colorbar.make_axes(ax)
    #cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f')

    plt.show()
