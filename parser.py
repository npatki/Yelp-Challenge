#!/usr/local/bin/python
import json
import numpy as np
import scipy as sp
import pprint

def parseData():
    # initialize arrays for machine learning
    featureVector = []
    cities = []
    ratings = []
    IDs = []

    # set the size of the feature vector
    featureSize = 19

    # counter for number of restauraunts
    count = 0

    #FIXME: determine what to do if data is not present
    # At the moment, all features are initialized with 0 so non-present
    # data gets a 0

    # cycle through all lines in the JSON yelp business dataset
    with open('../Data/yelp_academic_dataset_business.json') as f:
        for line in f:

            # load JSON data    
            rawData = json.loads(line)

            # check if the business is a restaurant
            if 'Restaurants' in rawData['categories']:

                # uncomment to print a lines
                #pprint.pprint(rawData, width=1)

                # save busineess ID
                ID = rawData['business_id']
                IDs.append(ID)

                # extract city
                city = rawData['city']
                cities.append(city)

                # extract rating
                rating = rawData['stars']
                ratings.append(rating)

                # allocate feature vector & extract features
                featureVector.append(np.zeros(featureSize))

                # feature 1 = review count
                numRatings = rawData['review_count']
                featureVector[count][0] = numRatings

                # features based on attributes
                attributes = rawData['attributes']

                # feature 2 = accepts credit cards
                if 'Accepts Credit Cards' in attributes:
                    featureVector[count][1] = \
                        binaryFeature(attributes['Accepts Credit Cards'])

                # features 3 - 11 = ambience (binary features)
                if 'Ambience' in attributes:
                    ambience = attributes['Ambience']

                    # feature 3 = casual
                    if 'casual' in ambience:
                        featureVector[count][2] = \
                                binaryFeature(ambience['casual'])

                    # feature 4 = classy
                    if 'classy' in ambience:
                        featureVector[count][3] = \
                                binaryFeature(ambience['classy'])

                    # feature 5 = divey
                    if 'divey' in ambience:
                        featureVector[count][4] = \
                                binaryFeature(ambience['divey'])

                    # feature 6 = hipster
                    if 'hipster' in ambience:
                        featureVector[count][5] = \
                                binaryFeature(ambience['hipster'])
                  
                    # feature 7 = intimate
                    if 'intimate' in ambience:
                        featureVector[count][6] = \
                                binaryFeature(ambience['intimate'])
                   
                    # feature 8 = romantic
                    if 'romantic' in ambience:
                        featureVector[count][7] = \
                                binaryFeature(ambience['romantic'])

                    # feature 9 = touristy
                    if 'touristy' in ambience:
                        featureVector[count][8] = \
                                binaryFeature(ambience['touristy'])

                    # feature 10 = trendy
                    if 'trendy' in ambience:
                        featureVector[count][9] = \
                                binaryFeature(ambience['trendy'])

                    # feature 11 = upscale
                    if 'upscale' in ambience:
                        featureVector[count][10] = \
                                binaryFeature(ambience['upscale'])
                   
                # feature 12 = alcohol offered
                if 'Alcohol' in attributes:
                    if attributes['Alcohol'] == 'none':
                        featureVector[count][11] = 0
                    else:
                        featureVector[count][11] = 1

                # feature 13 = casual attire
                if 'Attire' in attributes:
                    if attributes['Attire'] == 'casual':
                        featureVector[count][12] = 1
                    else:
                        featureVector[count][12] = 0

                # features 14 - 19 = what restauarant is good for
                if 'Good For' in attributes:
                    purpose = attributes['Good For']
                    
                    # feature 14 = good for breakfast
                    if 'breakfast' in purpose:
                        featureVector[count][13] = \
                            binaryFeature(purpose['breakfast'])

                    # feature 15 = good for brunch
                    if 'brunch' in purpose:
                        featureVector[count][14] = \
                            binaryFeature(purpose['brunch'])

                    # feature 16 = good for dessert
                    if 'dessert' in purpose:
                        featureVector[count][15] = \
                            binaryFeature(purpose['dessert'])

                    # feature 17 = good for dinner
                    if 'dinner' in purpose:
                        featureVector[count][16] = \
                            binaryFeature(purpose['dinner'])

                    # feature 18 = good for late night
                    if 'latenight' in purpose:
                        featureVector[count][17] = \
                            binaryFeature(purpose['latenight'])
                    
                    # feature 19 = good for lunch
                    if 'lunch' in purpose:
                        featureVector[count][17] = \
                            binaryFeature(purpose['lunch'])

                # increment the number of restaurants found
                count += 1

    return (ID, cities, ratings, featureVector)


def binaryFeature(binaryVal):
    if binaryVal:
        return 1
    else:
        return 0

def runScript():
    (IDs, cities, ratings, featureVector) = parseData()
    print cities

if __name__ == "__main__":
    runScript()
