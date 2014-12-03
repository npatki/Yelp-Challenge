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

# assign number of partitions for business and user data
num_business_partitions = 4
num_user_partitions = 4

"""
A function that loads all user information into a dictionary. Information is 
stored as (user_id, [review_count, average_stars])
"""
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

"""
A function that loads business information from .csv files into a dictionary. 
Information is stored as (business_id, business_types)  where business_types is
a vector of all applicable business types to the particular business.
"""
def LoadBusinessInformation(business_file, biz_types):

    print "Loading business dictionary..."
    
    # initialize array of indexes into CSV file
    type_indexes = range(len(biz_types))

    # create dicitonary of businesses by business ID
    businessDict = dict()

    # open business file and cycle through all businesses
    with open(business_file, 'rb') as csvfile:

        # read all lines in the file
        lines = csvfile.readlines()

        # read the header, find indexes for biz types
        header = lines[0].split()
        for i,word in enumerate(header):
            for j,typ in enumerate(biz_types):
                if word == typ:
                    type_indexes[j] = i

        # cycle through bulk data and make sets of biz type for businesses
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
                    types.append(biz_types[i])

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

    print "Loading review dictionary..."

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
    business_file_base = '../data/nl_biz_features_'
    user_file = '../data/yelp_academic_dataset_user.json'
    review_file = '../data/yelp_academic_dataset_review.json'

    # list of business types
    biz_types = [
        'Wine Bars','Jazz & Blues', 'Gay Bars', 'American (Traditional)',
        'Gay Bars', 'Breweries', 'Karaoke', 'Dive Bars', 'Restaurants',
        'Bars', 'Lounges', 'Dance Clubs', 'Sports Bars', 'Pubs', 
        'Music Venues']
    
    # put biz types into dictionary
    ntypes = len(biz_types)
    typeDict = dict( zip(biz_types, range(ntypes)) )

    # load user dicitonaries
    userDict = LoadUserInformation(user_file)
    reviewDict = LoadReviewInformation(review_file)

    # load business dicitonary 
    businessDict = dict()
    for n in range(num_business_partitions):
        business_file = business_file_base + str(n) + ".csv"
        businessDict.update( \
                LoadBusinessInformation(business_file, biz_types) )

    # create header and define list of lines to output to files
    outputLines = []
    header = ['user_id']
    for i in xrange(ntypes):
        name = biz_types[i] + '_num'
        header.append(name)
    for i in xrange(ntypes):
        name = biz_types[i] + '_rating'
        header.append(name)
    header.append('sample_reviews')
    header.append('sample_variance')
    header.append('sample_mean')
    header.append('total_reviews')
    header.append('total_mean')

    print "Forming feature vectors"

    # cycle through users, determine if they should be added to the dictionary
    for user in reviewDict:
        
        # create feature vector
        featureVector = np.zeros(2*ntypes + 5)

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
            featureVector[-5] = count

            # print feature vector to CSV file
            vector = [user]
            vector.extend(featureVector)
            
            # store data that will be output to file
            outputLines.append(vector)

    # calculate number of lines per file
    numLines = len(outputLines)/num_user_partitions

    # print output data to partioned files
    for n in range(num_user_partitions):
        writeLines = outputLines[n*numLines:(n+1)*numLines]
        with open("../data/user_features_%d.csv" % n, 'w') as out:
            out.write(' '.join((str(j) for j in header)) + '\n')
            for vector in writeLines:
                out.write(' '.join((str(j) for j in vector)) + '\n')
