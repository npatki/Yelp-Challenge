from solver import determineClusters

# If users are divided into x clusters, then this script
# writes x files where each represents the cluster. The file
# contains the user IDs of users belonging to the cluster. Each
# line contains a different user ID.
# 
# The file names are in the format: ../data/cluster_x.txt where
# x ranges from 0 to (NUM_CLUSTERS-1)

NUM_CLUSTERS = 6

if __name__ == '__main__':
    clusters = determineClusters('../data/user_features.csv',
        NUM_CLUSTERS, analyze_clusters=False)

    files = []
    for i in range(NUM_CLUSTERS):
        f = open('../data/cluster_%d.txt' % i, 'w')
        files.append(f)

    for user in clusters:
        cluster_num = clusters[user]
        files[cluster_num].write(user + '\n')

    for f in files:
        f.close()
