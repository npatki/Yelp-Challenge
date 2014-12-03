"""
A script that extracts features from businesses and writes a new file of feature
vectors.
"""
from copy import deepcopy as copy
import json
import math

BUSINESSES = [
    '../data/nl_biz_0.json',
    '../data/nl_biz_1.json',
    '../data/nl_biz_2.json',
    '../data/nl_biz_3.json'
]

OUT = [
    '../data/nl_biz_features_0.csv',
    '../data/nl_biz_features_1.csv',
    '../data/nl_biz_features_2.csv',
    '../data/nl_biz_features_3.csv'
]

# TODO: script that uses the extractors and puts the input vectors into a file
#       first element is ID, last element is rating 
# TODO: add new features like distance from center of city?
# TODO: user feature vector (with reviews >= 10)
#       ratings for each category
#       user ID, features -> [italian_avg, num_italian, ... num_reviews, variance]
# TODO: run classification on users --> feature vectors
# TODO: recalculate ratings + running the regression scripts
feature_names = [
    'ID',
    'casual_true', 'casual_false', 'casual_missing',
    'classy_true', 'classy_false', 'classy_missing',
    'divey_true', 'divey_false', 'divey_missing',
    'hipster_true', 'hipster_false', 'hipster_missing',
    'intimate_true', 'intimate_false', 'intimate_missing',
    'romantic_true', 'romantic_false', 'romantic_misssing',
    'toursity_true', 'touristy_false', 'touristy_missing',
    'trendy_true', 'trending_false', 'trendy_missing',
    'upscale_true', 'upscale_false', 'upscale_missing',
    'credit_true', 'credit_false', 'credit_missing',
    'delivery_true', 'delivery_false', 'delivery_missing',
    'kids_true', 'kids_false', 'kids_missing',
    'groups_true', 'groups_false', 'groups_missing',
    'tv_true', 'tv_false', 'tv_missing',
    'outdoor_true', 'outdoor_false', 'outdoor_missing',
    'waiter_true', 'waiter_false', 'waiter_missing',
    'full_bar', 'beer_and_wine', 'none', 'alcohol_missing',
    'Smoking', 'No_Smoking', 'Outdoor_Smoking', 'Smoking_missing',
    'casual', 'dressy', 'formal', 'attire_missing',
    '$', '$$', '$$$', '$$$$', 'price_range_missing',
    'Wine_Bars','Jazz', 'Gay_Bars', 'American_Traditional',
    'Breweries', 'Karaoke', 'Dive_Bars', 'Restaurants', 'Bars',
    'Lounges', 'Dance_Clubs', 'Sports_Bars', 'Pubs', 'Music_Venues',
    'phoenix', 'las_vegas', 'madison', 'waterloo', 'edinburgh',
    'distance','review_count', 'stars'
]
names_string = ' '.join(feature_names)

class AmbienceExtractor(object):

    all_ambiences = [
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
        correspond to the appropriate ambiences."""

        vector = []

        if 'Ambience' in data['attributes']:
            ambiences = data['attributes']['Ambience']

            for ambience in self.all_ambiences:
                if ambience in ambiences:
                    if ambiences[ambience]:
                        vector.extend(copy(self.vectors['true']))
                    else:
                        vector.extend(copy(self.vectors['false']))
                else:
                    vector.extend(copy(self.vectors['missing']))
        else:
            for i in xrange(len(self.all_ambiences)):
                vector.extend(copy(self.vectors['missing']))

        return vector


class BooleanAttributesExtractor(object):

    # attribute that have True/False values
    attributes = [
        'Accepts Credit Cards',
        'Coat Check',
        'Good for Kids',
        'Good For Groups',
        'Good For Dancing',
        'Has TV',
        'Outdoor Seating',
        'Waiter Service',
        'Happy Hour'
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
    noise_levels = [
            'quiet',
            'average',
            'loud',
            'very_loud'
    ]
    smoking = [
            'yes', 
            'no', 
            'outdoor'
    ]
    
    # also price ranges
    prices = [1,2,3,4]
 
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

        # add noise level features
        if 'Noise Level' in data['attributes']:
            for noise in self.noise_levels:
                if data['attributes']['Noise Level'] == noise:
                    vector.append(1)
                else:
                    vector.append(0)
            vector.append(0)
        else:
            for noise in self.noise_levels:
                vector.append(0)
            vector.append(1)

        # add smoking features
        if 'Smoking' in data['attributes']:
            for smoke in self.smoking:
                if data['attributes']['Smoking'] == smoke:
                    vector.append(1)
                else:
                    vector.append(0)
            vector.append(0)
        else:
            for smoke in self.smoking:
                vector.append(0)
            vector.append(1)


        # add price range features
        if 'Price Range' in data['attributes']:
            for price in self.prices:
                if data['attributes']['Price Range'] == price:
                    vector.append(1)
                else:
                    vector.append(0)
            vector.append(0)
        else:
            for price in self.prices:
                vector.append(0)
            vector.append(1)

        return vector


class CategoryExtractor(object):

    # These are all the categories that have >= 500 businesses
    # in the business data set in the 'Restaurants' category
    # We don't include 'Nightlife' because all data points include
    # that category.
    # list of business types
    all_categories = [
        'Wine Bars','Jazz & Blues', 'Gay Bars', 'American (Traditional)',
        'Breweries', 'Karaoke', 'Dive Bars', 'Restaurants', 'Bars',
        'Lounges', 'Dance Clubs', 'Sports Bars', 'Pubs', 'Music Venues']
    
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
        AmbienceExtractor(),
        BooleanAttributesExtractor(),
        StringAttributesExtractor(),
        CategoryExtractor(),
        CityExtractor(),
        RatingExtractor
    ]

    for i, BIZ in enumerate(BUSINESSES):
        out = open(OUT[i], 'w')
        out.write(names_string + '\n')

        with open(BIZ) as f:
            for line in f:
                data = json.loads(line)
                vector = [data['business_id']]

                for extractor in extractors:
                    vector.extend(extractor(data))
                out.write(' '.join((str(j) for j in vector)) + '\n')
        out.close()
