"""
A script that extracts features from user and writes a new file of feature
vectors.
"""
from copy import deepcopy as copy
import json
import csv
import numpy as np
from math import *

# assign cut off for reviews
user_min_reviews = 10


def LoadUserInformation(user_file):
   
    print "Loading user dictionary..."

    # create dicitonary of user averages & number of total reviews
    userDict = dict()

    # cycle through user file to load information into dictionary
    with open(user_file) as f:
        for line in f:

            # load JSON data line
            rawData = json.loads(line)

            # load user ID
            userID = rawData['user_id']

            # load user information
            userInfo = [ rawData['review_count'], rawData['average_stars'] ]

            # save information to dictionary
            userDict[userID] = userInfo

    return userDict

def LoadBusinessInformation(business_file, restaurant_types):

    print "Loading business dictionary..."
    
    # initialize array of indexes into CSV file
    type_indexes = range(len(restaurant_types))

    # create dicitonary of businesses by business ID
    businessDict = dict()

    # open business file and cycle through all businesses
    with open(business_file, 'rb') as csvfile:

        # read all lines in the file
        lines = csvfile.readlines()

        # read the header, find indexes for restaurant types
        header = lines[0].split()
        for i,word in enumerate(header):
            for j,typ in enumerate(restaurant_types):
                if word == typ:
                    type_indexes[j] = i

        # cycle through bulk data and make sets of restaurant type for businesses
        for line in lines[1:]:
        
            # split line into individual enteries
            words = line.split()
            
            # extract ID
            ID = words[0]
            
            # intialize list of types
            types = []

            # find which types are associated with the business
            for i,index in enumerate(type_indexes):
                if int(words[index]) == 1:
                    types.append(restaurant_types[i])

            # change types into a set
            typeSet = set(types)

            # add set to dicitonary
            businessDict[ID] = typeSet

    return businessDict

"""
A function that loads all review information into a dictionary. Information is
stored as (user_id, [business_id, stars])
"""
def LoadReviewInformation(review_file):

    print "Loading review dicitonary..."

    # initialize dictionary of user information
    reviewDict = dict()

    # cycle through business file to load information
    with open(review_file) as f:
        for line in f:

            # load JSON data line
            rawData = json.loads(line)

            # load user id
            userID = rawData['user_id']
            
            # add user to dictionary if not already added            
            if userID not in reviewDict:
                reviewDict.update( [(userID, [])] )

            # load user information, store as duple
            reviewInfo = (rawData['business_id'], rawData['stars'])

            # save information in dictionary
            reviewDict[userID].append(reviewInfo)
            
    return reviewDict

# run script
if __name__ == '__main__':

    # file names
    business_file_base = '../data/biz_features_'
    user_file = '../data/yelp_academic_dataset_user.json'
    review_file = '../data/yelp_academic_dataset_review.json'

    # list of restuarant types
    restaurant_types = [
        'Mexican','American_Traditional', 'Fast_Food',
        'Pizza', 'Sandwiches', 'Nightlife', 'Bars', 'Food',
        'American_New', 'Italian', 'Chinese', 'Burgers',
        'Breakfast_Brunch', 'Japanese']

    ntypes = len(restaurant_types)

    typeDict = dict( zip(restaurant_types, range(ntypes)) )

    # load user dicitonaries
    userDict = LoadUserInformation(user_file)
    reviewDict = LoadReviewInformation(review_file)

    # create user dataset for each business partition file
    for n in range(4):

        # load business dicitonary 
        business_file = business_file_base + str(n) + ".csv"
        businessDict = LoadBusinessInformation(business_file, restaurant_types)

        # open output file
        out = open("../data/user_features_" + str(n) + ".csv", 'w')

        print "Forming feature vectors"

        # cycle through users, determine if they should be added to the dictionary
        for user in reviewDict:
            
            # create feature vector
            featureVector = np.zeros(2*ntypes + 4)

            # count how many businesses present in the dataset
            count = 0
            S1 = 0
            S2 = 0

            # cycle through businesses
            for entry in reviewDict[user]:

                # determine if business present in business dictionary
                business = entry[0]
                rating = entry[1]
                
                # check if business is in the partition
                if business in businessDict:
                    
                    # add attributes
                    attributes = businessDict[business]
                    for typ in attributes:
                        index = typeDict[typ]
                        featureVector[index] += 1
                        featureVector[index + ntypes] += rating

                    # add count
                    count += 1
                    S1 += rating
                    S2 += rating**2

            # determine if user provided enough reviews
            if count >= user_min_reviews:
                
                # load user information
                numRatings = userDict[user][0]
                avgRating = userDict[user][1]
                
                # calculate set mean and variance
                mean = float(S1) / count
                variance = float(S2)/count - mean**2

                # correct freature vector
                for i in xrange(ntypes):
                    typeCount = featureVector[i]
                    if typeCount > 0:
                        featureVector[ntypes + i] /= typeCount
                        featureVector[ntypes + i] -= avgRating

                # add extra features
                featureVector[-1] = avgRating
                featureVector[-2] = numRatings
                featureVector[-3] = mean
                featureVector[-4] = variance

                # print feature vector to CSV file
                vector = [user]
                vector.extend(featureVector)
                out.write(' '.join((str(j) for j in vector)) + '\n')

        out.close()
