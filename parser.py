#!/usr/local/bin/python
from math import *
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
    locations = []

    # set the size of the feature vector
    featureSize = 21

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

                # extract city (not used)
                city = rawData['city']

                # extract rating
                rating = rawData['stars']
                ratings.append(rating)

                
                # extract coordinates (latitude, longitude)
                longitude = rawData['longitude']
                latitude = rawData['latitude']
                location = (latitude, longitude)
                locations.append(location)
                
                # determine metro area from location
                city = determineMetro(location)
                cities.append(city)
                
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
                        featureVector[count][11] = -1
                    else:
                        featureVector[count][11] = 1

                # feature 13 = casual attire
                if 'Attire' in attributes:
                    if attributes['Attire'] == 'casual':
                        featureVector[count][12] = 1
                    else:
                        featureVector[count][12] = -1

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
                        featureVector[count][18] = \
                            binaryFeature(purpose['lunch'])

                # feature 20 = restaurant has TV
                if 'Has TV' in attributes:
                    featureVector[count][19] = \
                            binaryFeature(attributes['Has TV'])

                # feature 21 = price range
                if 'Price Range' in attributes:
                    featureVector[count][20] = \
                            binaryFeature(attributes['Price Range'])
                
                
                # increment the number of restaurants found
                count += 1

    plotLocations(locations,cities)
    return (IDs, cities, ratings, featureVector)

def plotLocations(locations,cities):
    
    import matplotlib.pyplot as plt
    
    for i,loc in enumerate(locations):
        c = 'w'
        if cities[i] == "Phoenix":
            c = 'k'
        elif cities[i] == "Las Vegas":
            c = 'm'
        elif cities[i] == "Madison":
            c = 'r'
        elif cities[i] == "Waterloo":
            c = 'b'
        elif cities[i] == "Edinburgh":
            c = 'g'
        plt.plot(loc[1],loc[0],'.', color = c)
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

# convert a feature that is True/False into +1/-1
def binaryFeature(binaryVal):
    if binaryVal:
        return 1
    else:
        return -1

# determines metro area from (latitude, longitude) location coordintes
def determineMetro(loc):

    # distance within which a location is considered part of the metro area
    tolerance = 3

    # check for cities: Phoenix, Las Vegas, Madison, Waterloo, and Edinburgh
    if sqrt( (loc[0] - 33.27)**2 + (loc[1] + 112.04)**2 ) < tolerance:
        return "Phoenix"
    elif sqrt( (loc[0] - 36.10)**2 + (loc[1] + 115.08)**2 ) < tolerance:
        return "Las Vegas"
    elif sqrt( (loc[0] - 43.3)**2 + (loc[1] + 89.24)**2 ) < tolerance:
        return "Madison"
    elif sqrt( (loc[0] - 43.28)**2 + (loc[1] + 80.31)**2 ) < tolerance:
        return "Waterloo"
    elif sqrt( (loc[0] - 55.57)**2 + (loc[1] + 3.11)**2 ) < tolerance:
        return "Edinburgh"
    else:
        print "Error: Could not determine city from coordinates"
        print loc
        return ""

# executes when script is run
def runScript():
    (IDs, cities, ratings, featureVector) = parseData()
    print cities

if __name__ == "__main__":
    runScript()
