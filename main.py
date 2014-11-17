#!/usr/local/bin/python
from math import *
import json
import numpy as np
import scipy as sp
import pprint
from extractor import *
#from partitioner import *


def main():

    # set paths to business, user, and review information
    businessFile = '../Data/yelp_academic_dataset_business.json'
    userFile = '../Data/yelp_academic_dataset_user.json'
    reviewFile = '../Data/yelp_academic_dataset_review.json'

    # load business dictionary (business_id keys, feature vector values)
    print "Loading business information..."
    businessDict = LoadBusinessInformation(businessFile)

    # load user dictionary (user_id keys, [review_count, average_stars] values)
    print "Loading user information..."
    userDict = LoadUserInformation(userFile)

    # determine how many users have more than 10 reviews, average number of reviews
    print "\nDetermining user breakdown:"
    nusers = 0
    rev10 = 0
    revtotal = 0
    for user in userDict.keys():
        nusers += 1
        numrevs = userDict[user][0]
        revtotal += numrevs
        if numrevs >= 10:
            rev10 +=  1
    print "More than 10 reviews = " + str(rev10) + "/" + str(nusers) + " users"
    print "Average review number = " + str( float(revtotal) / float(nusers) ) + '\n'

    # load review dictionary (user_id keys, [business_id, stars] values)
    print "Loading review information..."
    reviewDict = LoadReviewInformation(reviewFile)

    # determine how many users have more than 10 reviews, average number of reviews
    print "\nDetermining user review breakdown:"
    nusers = 0
    revX = 0
    revtotal = 0
    for user in reviewDict.keys():
        nusers += 1
        numrevs = len(reviewDict[user])
        revtotal += numrevs
        if numrevs >= 10:
            revX +=  1
 
    print "More than 10 reviews = " + str(revX) + "/" + str(nusers) + " users"
    print "Average review number = " + str( float(revtotal) / float(nusers) ) + '\n'


    # create new dictionary that stores relevant (restaurant) information
    print "\nRestaurant review breakdown:"
    nusers = 0
    revX = 0
    revtotal = 0
    for user in reviewDict.keys():
        numrevs = 0
        reviews = reviewDict[user]
        for review in reviews:
            business = review[0] 
            if business in businessDict:
                numrevs += 1
        if numrevs > 0:
            nusers += 1
            revtotal += numrevs
            if numrevs >= 10:
                revX += 1

    print "More than 10 restaurant reviews = " + str(revX) + "/" + str(nusers) + " users"
    print "Average restaurant review number = " + str( float(revtotal) / float(nusers) ) + '\n'


"""
A function that loads all business information into a feature vector for
training. The function returns a duple of (business ID, feature vector).
"""
def LoadBusinessInformation(filename):

    # initialize business dictionary
    businessDict = dict()

    # initialize extractors
    ratingExtract = RatingExtractor
    cityExtract = CityExtractor()
    categoryExtract = CategoryExtractor()
    stringFeatureExtract = StringAttributesExtractor()
    booleanFeatureExtract = BooleanAttributesExtractor()
    goodForExtract = GoodForExtractor()
    ambianceExtract = AmbianceExtractor()


    # cycle through business file to load information
    with open(filename) as f:
        for line in f:

            # load JSON data line
            rawData = json.loads(line)

            # only select restaurants
            if 'Restaurants' in rawData['categories']:

                # record business ID
                #IDs.append( rawData['business_id'] )

                # extract rating information
                ratingInfo = ratingExtract(rawData)

                # extract city
                (city, cityCode) = cityExtract(rawData)

                # extract categories
                categories = categoryExtract(rawData)

                # extract string features
                stringFeatures = stringFeatureExtract(rawData)

                # extract boolean features
                booleanFeatures = booleanFeatureExtract(rawData)

                # extract what a business is "good for"
                goodFor = goodForExtract(rawData)

                # extract ambiance
                ambiance = ambianceExtract(rawData)

                # form feature vector
                featureVector = ratingInfo + cityCode + categories + \
                        stringFeatures + booleanFeatures + goodFor + ambiance
                
                # append feature vector to featureVectors list
                #featureVectors.append(featureVector)
                ID = rawData['business_id']
                businessDict.update( [(ID, featureVector)] )
    
    return businessDict


"""
A function that loads all user information into a dictionary. Information is
stored as (user_id, [review_count, average_stars])
"""
def LoadUserInformation(filename):

    # initialize dictionary of user information
    userDict = dict()

    # cycle through business file to load information
    with open(filename) as f:
        for line in f:

            # load JSON data line
            rawData = json.loads(line)

            # load user id
            userID = rawData['user_id']
            
            # load user information, store as duple
            userInfo = (rawData['review_count'], rawData['average_stars'])

            # save information in dictionary
            userDict.update( [(userID, userInfo)] )

    return userDict


"""
A function that loads all review information into a dictionary. Information is
stored as (user_id, [business_id, stars])
"""
def LoadReviewInformation(filename):

    # initialize dictionary of user information
    userDict = dict()

    # cycle through business file to load information
    with open(filename) as f:
        for line in f:

            # load JSON data line
            rawData = json.loads(line)

            # load user id
            userID = rawData['user_id']
            
            # add user to dictionary if not already added            
            if userID not in userDict:
                userDict.update( [(userID, [])] )

            # load user information, store as duple
            reviewInfo = (rawData['business_id'], rawData['stars'])

            # save information in dictionary
            userDict[userID].append(reviewInfo)
            
    return userDict



if __name__ == "__main__":
    main()


