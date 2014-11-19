"""
A script that extracts features from businesses and writes a new file of feature
vectors.
"""
from copy import deepcopy as copy
import json
import math

BUSINESSES = [
    '../data/biz_0.json',
    '../data/biz_1.json',
    '../data/biz_2.json',
    '../data/biz_3.json'
]

OUT = [
    '../data/biz_features_0.json',
    '../data/biz_features_1.json',
    '../data/biz_features_2.json',
    '../data/biz_features_3.json'
]

# TODO: script that uses the extractors and puts the input vectors into a file
#       first element is ID, last element is rating 
# TODO: add new features like distance from center of city?
# TODO: user feature vector (with reviews >= 10)
#       ratings for each category
#       user ID, features -> [italian_avg, num_italian, ... num_reviews, variance]
# TODO: run classification on users --> feature vectors
# TODO: recalculate ratings + running the regression scripts

class AmbianceExtractor(object):

    all_ambiances = [
        'casual',
        'classy',
        'divey',
        'hipster',
        'intimate',
        'romantic',
        'touristy',
        'trendy',
        'upscale'
    ]

    # deal with missing values by creating a binary
    # vector to represent true, false, missing
    vectors = {
        'true': [1, 0, 0],
        'false': [0, 1, 0],
        'missing': [0, 0, 1]
    }
    
    def __call__(self, data):
        """Return binary feature vector with 1's that
        correspond to the appropriate ambiances."""

        vector = []

        if 'Ambiance' in data['attributes']:
            ambiances = data['attributes']['Ambiance']

            for ambiance in self.all_ambiances:
                if ambiance in ambiances:
                    if ambiances[ambiance]:
                        vector.extend(copy(self.vectors['true']))
                    else:
                        vector.extend(copy(self.vectors['false']))
                else:
                    vector.extend(copy(self.vectors['missing']))
        else:
            for i in xrange(len(self.all_ambiances)):
                vector.extend(copy(self.vectors['missing']))

        return vector


class GoodForExtractor(object):

    all_purposes = [
        'breakfast',
        'brunch',
        'dessert',
        'dinner',
        'latenight',
        'lunch'
    ]

    # deal with missing values by creating a binary
    # vector to represent true, false, missing
    vectors = {
        'true': [1, 0, 0],
        'false': [0, 1, 0],
        'missing': [0, 0, 1]
    }
    
    def __call__(self, data):
        """Return binary feature vector with 1's that
        correspond to the appropriate ambiances."""

        vector = []

        if 'Good For' in data['attributes']:
            purposes = data['attributes']['Good For']

            for purpose in self.all_purposes:
                if purpose in purposes:
                    if purposes[purpose]:
                        vector.extend(copy(self.vectors['true']))
                    else:
                        vector.extend(copy(self.vectors['false']))
                else:
                    vector.extend(copy(self.vectors['missing']))
        else:
            for i in xrange(len(self.all_purposes)):
                vector.extend(copy(self.vectors['missing']))

        return vector


class BooleanAttributesExtractor(object):

    # attribute that have True/False values
    attributes = [
        'Accepts Credit Cards',
        'Delivery',
        'Dogs Allowed',
        'Good for Kids',
        'Good For Groups',
        'Has TV',
        'Outdoor Seating',
        'Waiter Service'
    ]


    # deal with missing values by creating a binary
    # vector to represent true, false, missing
    vectors = {
        'true': [1, 0, 0],
        'false': [0, 1, 0],
        'missing': [0, 0, 1]
    }

    def __call__(self, data):
        vector = []

        for index, attribute in enumerate(self.attributes):
            if attribute in data['attributes']:
                if data['attributes'][attribute]:
                    vector.extend(copy(self.vectors['true']))
                else:
                    vector.extend(copy(self.vectors['false']))
            else:
                vector.extend(copy(self.vectors['missing']))

        return vector


class StringAttributesExtractor(object):

    # attributes that have string values
    alcohols = [
        'full_bar',
        'beer_and_wine',
        'none'
    ]

    attires = [
        'casual',
        'dressy',
        'formal'
    ]
 
    def __call__(self, data):
        vector = []

        # add alcohol features
        if 'Alcohol' in data['attributes']:
            for alcohol in self.alcohols:
                if data['attributes']['Alcohol'] == alcohol:
                    vector.append(1)
                else:
                    vector.append(0)
            vector.append(0)
        else:
            for alcohol in self.alcohols:
                vector.append(0)
            vector.append(1)

        # add attire features
        if 'Attire' in data['attributes']:
            for attire in self.attires:
                if data['attributes']['Attire'] == attire:
                    vector.append(1)
                else:
                    vector.append(0)
            vector.append(0)
        else:
            for attire in self.attires:
                vector.append(0)
            vector.append(1)

        return vector


class CategoryExtractor(object):

    # These are all the categories that have >= 500 businesses
    # in the business data set in the 'Restaurants' category
    # We don't include 'Restaurants' because all data points include
    # that category.
    all_categories = [
        'Mexican',
        'American (Traditional)',
        'Fast Food',
        'Pizza',
        'Sandwiches',
        'Nightlife',
        'Bars',
        'Food',
        'American (New)',
        'Italian',
        'Chinese',
        'Burgers',
        'Breakfast & Brunch',
        'Japanese'
    ]

    def __call__(self, data):
        """Return binary feature vector with 1's that 
        corresponds to the appropriate categories."""

        vector = [0]*len(self.all_categories)
        
        if 'categories' in data:
            categories = data['categories']
            for category in categories:
                try:
                    index = self.all_categories.index(category)
                except ValueError:
                    continue
                vector[index] = 1

        return vector


class CityExtractor(object):

    # Latitude, longitude locations for the 5 cities, as
    # provided by Google. Note that + corresponds to North/East,
    # - corresponds to South/West
    locations = {
        'Phoenix': (33.4500, -112.0667),
        'Las Vegas': (36.1215, -115.1739),
        'Madison': (43.0667, -89.4000),
        'Waterloo': (43.4667, -80.5167),
        'Edinburgh': (55.9531, -3.1889)
    }

    cityCodes = {
        'Phoenix': [1,0,0,0,0],
        'Las Vegas': [0,1,0,0,0],
        'Madison': [0,0,1,0,0],
        'Waterloo': [0,0,0,1,0],
        'Edinburgh': [0,0,0,0,1]
    }

    def __call__(self, data):
        loc = (data['latitude'], data['longitude'])

        best_guess_city = None
        distance_guess = float('inf')

        for city, center in self.locations.items():
            val = (center[0]-loc[0])**2 + (center[1]-loc[1])**2
            if val < distance_guess:
                distance_guess = val
                best_guess_city = city

        vector = copy(self.cityCodes[best_guess_city])
        vector.append(distance_guess)
        return vector


def RatingExtractor(data):
    return [data['review_count'], data['stars']]


if __name__ == '__main__':
    extractors = [
        AmbianceExtractor(),
        GoodForExtractor(),
        BooleanAttributesExtractor(),
        StringAttributesExtractor(),
        CategoryExtractor(),
        CityExtractor(),
        RatingExtractor
    ]

    for i, BIZ in enumerate(BUSINESSES):
        out = open(OUT[i], 'a+')

        with open(BIZ) as f:
            for line in f:
                data = json.loads(line)
                vector = []

                for extractor in extractors:
                    vector.extend(extractor(data))
                out.write(' '.join((str(j) for j in vector)) + '\n')
        out.close()
