"""
This is a script that splits the business data set randomly into
4 partitions. We only consider businesses that include 'Restaurants' in
their category list.

There are 14303 such businesses in the data set so we put 3575 or 3576
businesses in each partition.
"""
import json
import random

BUSINESS = '../data/yelp_academic_dataset_business.json'
MAX = 3576
OUT = [
    open('../data/biz_0.json', 'w'),
    open('../data/biz_1.json', 'w'),
    open('../data/biz_2.json', 'w'),
    open('../data/biz_3.json', 'w')
]

if __name__ == '__main__':
    # make it deterministic
    random.seed(0)

    # keep track of how many businesses we've assigned to each partition
    counts = [0, 0, 0, 0]

    with open(BUSINESS) as f:
        for line in f:
            data = json.loads(line)

            if 'Restaurants' in data['categories']:
                partition = random.randint(0, len(OUT)-1)

                # choose a partition that has spots available
                while counts[partition] >= MAX:
                    partition = random.randint(0, len(OUT)-1)

                OUT[partition].write(line)
                counts[partition] += 1

    for f in OUT:
        f.close()
