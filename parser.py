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
    featureSize = 65

    # counter for number of restauraunts
    count = 0

    # TODO: determine what to do if data is not present
    # At the moment, all features are initialized with 0 so non-present
    # data gets a 0

    # cycle through all lines in the JSON yelp business dataset
    with open('../Data/yelp_academic_dataset_user.json') as f:
        for line in f:

            # load JSON data    
            rawData = json.loads(line)

            pprint.pprint(rawData, width=1)
            exit()

            # check if the business is a restaurant
            if 'Restaurants' in rawData['categories']:

                # uncomment to print a line
                #pprint.pprint(rawData, width=1)
                #exit()

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

                # feature (2,3,4) = accepts credit cards
                if 'Accepts Credit Cards' in attributes:
                    if attributes['Accepts Credit Cards']:
                        featureVector[count][1] = 1
                    else:
                        featureVector[count][2] = 1
                else:
                    featureVector[count][3] = 1

                # features (5,6,7) - (29,30,31) = ambience (binary features)
                if 'Ambience' in attributes:
                    ambience = attributes['Ambience']

                    # feature (5,6,7) = casual
                    if 'casual' in ambience:
                        if ambience['casual']:
                            featureVector[count][4] = 1
                        else:
                            featureVector[count][5] = 1
                    else:
                        featureVector[count][6] = 1

                    # feature (8,9,10) = classy
                    if 'classy' in ambience:
                        if ambience['classy']:
                            featureVector[count][7] = 1
                        else:
                            featureVector[count][8] = 1
                    else:
                        featureVector[count][9] = 1

                    # feature (11,12,13) = divey
                    if 'divey' in ambience:
                        if ambience['divey']:
                            featureVector[count][10] = 1
                        else:
                            featureVector[count][11] = 1
                    else:
                        featureVector[count][12] = 1

                    # feature (14,15,16) = hipster
                    if 'hipster' in ambience:
                        if ambience['hipster']:
                            featureVector[count][13] = 1
                        else:
                            featureVector[count][14] = 1
                    else:
                        featureVector[count][15] = 1
                  
                    # feature (17,18,19) = intimate
                    if 'intimate' in ambience:
                        if ambience['intimate']:
                            featureVector[count][16] = 1
                        else:
                            featureVector[count][17] = 1
                    else:
                        featureVector[count][18] = 1
                   
                    # feature (20,21,22) = romantic
                    if 'romantic' in ambience:
                        if ambience['romantic']:
                            featureVector[count][19] = 1
                        else:
                            featureVector[count][20] = 1
                    else:
                        featureVector[count][21] = 1

                    # feature (23,24,25) = touristy
                    if 'touristy' in ambience:
                        if ambience['touristy']:
                            featureVector[count][22] = 1
                        else:
                            featureVector[count][23] = 1
                    else:
                        featureVector[count][24] = 1

                    # feature (26,27,28) = trendy
                    if 'trendy' in ambience:
                        if ambience['trendy']:
                            featureVector[count][25] = 1
                        else:
                            featureVector[count][26] = 1
                    else:
                        featureVector[count][27] = 1

                    # feature (29,30,31) = upscale
                    if 'upscale' in ambience:
                        if ambience['upscale']:
                            featureVector[count][28] = 1
                        else:
                            featureVector[count][29] = 1
                    else:
                        featureVector[count][30] = 1
                
                else:
                    # fill features with "unknown" marker
                    featureVector[count][6] = 1
                    featureVector[count][9] = 1
                    featureVector[count][12] = 1
                    featureVector[count][15] = 1
                    featureVector[count][18] = 1
                    featureVector[count][21] = 1
                    featureVector[count][24] = 1
                    featureVector[count][27] = 1
                    featureVector[count][30] = 1

                # feature (32,33,34) = alcohol offered
                if 'Alcohol' in attributes:
                    if attributes['Alcohol'] == 'none':
                        featureVector[count][31] = 1
                    else:
                        featureVector[count][32] = 1
                else:
                    featureVector[count][33] = 1

                # feature (35,36,37) = casual attire
                if 'Attire' in attributes:
                    if attributes['Attire'] == 'casual':
                        featureVector[count][34] = 1
                    else:
                        featureVector[count][35] = 1
                else:
                    featureVector[count][36] = 1

                # features (38-40) - (53-55) = what restauarant is good for
                if 'Good For' in attributes:
                    purpose = attributes['Good For']
                    
                    # feature (38,39,40) = good for breakfast
                    if 'breakfast' in purpose:
                        if purpose['breakfast']:
                            featureVector[count][37] = 1
                        else:
                            featureVector[count][38] = 1
                    else:
                        featureVector[count][39] = 1

                    # feature (41,42,43) = good for brunch
                    if 'brunch' in purpose:
                        if purpose['brunch']:
                            featureVector[count][40] = 1
                        else:
                            featureVector[count][41] = 1
                    else:
                        featureVector[count][42] = 1

                    # feature (44,45,46) = good for dessert
                    if 'dessert' in purpose:
                        if purpose['dessert']:
                            featureVector[count][43] = 1
                        else:
                            featureVector[count][44] = 1
                    else:
                        featureVector[count][45] = 1

                    # feature (47,48,49) = good for dinner
                    if 'dinner' in purpose:
                        if purpose['dinner']:
                            featureVector[count][46] = 1
                        else:
                            featureVector[count][47] = 1
                    else:
                        featureVector[count][48] = 1

                    # feature (50,51,52) = good for late night
                    if 'latenight' in purpose:
                        if purpose['latenight']:
                            featureVector[count][49] = 1
                        else:
                            featureVector[count][50] = 1
                    else:
                        featureVector[count][51] = 1
                    
                    # feature (53,54,55) = good for lunch
                    if 'lunch' in purpose:
                        if purpose['lunch']:
                            featureVector[count][52] = 1
                        else:
                            featureVector[count][53] = 1
                    else:
                        featureVector[count][54] = 1
                else:
                    # fill feature vector with "unknown" markers
                    featureVector[count][39] = 1
                    featureVector[count][42] = 1
                    featureVector[count][45] = 1
                    featureVector[count][48] = 1
                    featureVector[count][51] = 1
                    featureVector[count][54] = 1
                    

                # feature (56,57,58) = restaurant has TV
                if 'Has TV' in attributes:
                    if attributes['Has TV']:
                        featureVector[count][55] = 1
                    else:
                        featureVector[count][56] = 1
                else:
                    featureVector[count][57] = 1

                # feature (59-64) = price range
                if 'Price Range' in attributes:
                    featureVector[count][58+attributes['Price Range']] = 1
                else:
                    featureVector[count][58] = 1
                
                # add rating to feature vector
                featureVector[count][64] = rating

                # increment the number of restaurants found
                count += 1

    return (IDs, cities, locations, ratings, featureVector)

# creates a plot of business locations by (latitude, longitude) coordinates
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
    (IDs, cities, locations, ratings, featureVector) = parseData()
    plotLocations(locations,cities)

if __name__ == "__main__":
    runScript()
