#!/bin/python 

import numpy
import os
from sklearn.cluster.k_means_ import KMeans
import pickle
import sys
import csv
from sklearn.cluster import MiniBatchKMeans

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0]))
        print("mfcc_csv_file -- path to the mfcc csv file")
        print("cluster_num -- number of cluster")
        print("output_file -- path to save the k-means model")
        exit(1)

    print("train kmeans started")

    mfcc_csv_file = sys.argv[1]
    output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])

    ip_file = open(mfcc_csv_file, 'r')

    csv_list = []

    for i, line in enumerate(ip_file):
        if i % 2 == 0:
            continue
        if i % 1000000 == 0:
            print(i)
        each_list = [float(l.strip()) for l in line.split(';')]
        csv_list.append(each_list)

    print(len(csv_list))
    print(len(csv_list[0]))

    kmeans = MiniBatchKMeans(n_clusters=cluster_num, random_state=0, batch_size=100).fit(csv_list)
    pickle.dump(kmeans, open(output_file, 'wb'))

    print("K-means trained successfully!")
