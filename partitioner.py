"""
This is a script that splits the business data set randomly into
4 partitions. There are a total of 42153 business data points,
so we parition into sets of approximately 10538. It's deterministic.
"""

import random

BUSINESS = '../data/yelp_academic_dataset_business.json'
MAX = 10538
OUT = [
    open('../data/biz_0.json', 'a+'),
    open('../data/biz_1.json', 'a+'),
    open('../data/biz_2.json', 'a+'),
    open('../data/biz_3.json', 'a+')
]

if __name__ == '__main__':
    first = False

    # make it deterministic
    random.seed(0)
    counts = [0, 0, 0, 0]

    with open(BUSINESS) as f:
        for line in f:
            if not first:
                OUT[0].write(line)
                first = True
                continue

            partition = random.randint(0, len(OUT)-1)
            while counts[partition] >= MAX:
                partition = random.randint(0, len(OUT)-1)

            OUT[partition].write(line)
            counts[partition] += 1

    for f in OUT:
        f.close()
