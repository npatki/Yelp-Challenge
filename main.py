#!/usr/local/bin/python
from math import *
import json
import numpy as np
import scipy as sp
import pprint
from extractor import *
#from partitioner import *


"""
A function that loads all business information into a feature vector for
training. The function returns a duple of (business ID, feature vector).
"""
def LoadBusinessInformation(filename):

    # initialize feature vector
    featureVectors = []
    IDs = []

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
                IDs.append( rawData['business_id'] )

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
                featureVectors.append(featureVector)
    
    return IDs, featureVectors

def main():

    # set paths to business, user, and review information
    businessFile = '../Data/yelp_academic_dataset_business.json'
    userFile = '../Data/yelp_academic_dataset_user.json'
    reviewFile = '../Data/yelp_academic_dataset_review.json'

    # load business IDs, feature vectors
    (businessIDs, businessFeatures) = LoadBusinessInformation(businessFile)

    print len(businessIDs)


if __name__ == "__main__":
    main()
