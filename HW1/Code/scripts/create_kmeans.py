#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]
    file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    fread = open(file_list,"r")

    print("create kmeans started")

    # load the kmeans model
    video_vector_map = {}
    kmeans = cPickle.load(open(kmeans_model,"rb"))
    final_vector = []
    for line in fread.readlines():
        print("create kmeans")
        print(str(line))
        mfcc_path = "mfcc/" + line.replace('\n','') + ".mfcc.csv"
        if os.path.exists(mfcc_path) == True:
            array = numpy.genfromtxt(mfcc_path, delimiter=";")
            cluster_numbers = kmeans.predict(array)
            video_vector = []
            cluster_count_map = {}
            for cluster_number in cluster_numbers:
                if cluster_number not in cluster_count_map:
                    cluster_count_map[cluster_number] = 0
                cluster_count_map[cluster_number] += 1
            for cluster in range(cluster_num):
                if cluster in cluster_count_map:
                    video_vector.append(cluster_count_map[cluster])
                else:
                    video_vector.append(0)
            video_vector = ((numpy.array(video_vector) * 1.0) / numpy.sum(video_vector)).tolist()
        else:
            video_vector = [1.0 / cluster_num] * cluster_num
        video_vector_map[str(line)] = video_vector

    output_file = "/project/hsr/sshalini/LSMA/11775-hws/hw1_code/kmeans/features_0.5"
    cPickle.dump(video_vector_map, open(output_file, 'w+'))



    print "K-means features generated successfully!"
