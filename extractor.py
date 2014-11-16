"""
A script that extracts features and writes a new file of feature
vectors.
"""
from copy import deepcopy as copy
import math


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


class CategoryExtractor(object):

    # These are all the categories that have >= 1000 businesses
    # in the business data set
    all_categories = [
        'Restaurants',
        'Shopping',
        'Food',
        'Beauty & Spas',
        'Nightlife',
        'Bars',
        'Health & Medical',
        'Automotive',
        'Home Services',
        'Fashion',
        'Active Life',
        'Mexican',
        'Event Planning & Services',
        'American (Traditional)',
        'Fast Food',
        'Local Services',
        'Pizza',
        'Hotels & Travel',
        'Arts & Entertainment',
        'Sandwiches',
        'Coffee & Tea',
        'American (New)',
        'Italian',
        'Chinese'
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

    def __call__(self, data):
        loc = (data['latitude'], data['longitude'])

        best_guess_city = None
        distance_guess = float('inf')

        for city, center in self.location.items():
            val = (center[0]-loc[0])**2 + (center[1]-loc[1])**2
            if val < distance_guess:
                distance_guess = val
                best_guess_city = city

        return best_guess_city


def rating_extractor(data):
    return [data['review_count'], data['stars']]
