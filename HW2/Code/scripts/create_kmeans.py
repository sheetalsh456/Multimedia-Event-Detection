#!/bin/python
import numpy
import os
import pickle
from sklearn.cluster.k_means_ import KMeans
from joblib import Parallel, delayed
import sys
# Generate k-means features for videos; each video is represented by a single vector
# 
# python create_kmeans.py kmeans_surf.model 400 list/all.video

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0]))
        print("kmeans_model -- path to the kmeans model")
        print("cluster_num -- number of cluster")
        print("file_list -- the list of videos")
        exit(1)

    kmeans_model = sys.argv[1]
    file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    fread = open(file_list,"r")
    lines = fread.readlines()

    print("create kmeans started")

    video_vector_map = {}
    kmeans = pickle.load(open(kmeans_model,"rb"))

    def create(i, line):
        print("create kmeans")
        line = line.strip()
        print(i)
        print(str(line))
        surf_path = "surf/" + line.replace('\n','') + ".surf"
        if os.path.exists(surf_path) == True:
            keyframes = pickle.load(open(surf_path, 'rb'))
            keyframes_new = [frame for frame in keyframes if frame is not None]
            try:
                array = numpy.vstack(keyframes_new)
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
            except:
                video_vector = [1.0 / cluster_num] * cluster_num
        else:
            video_vector = [1.0 / cluster_num] * cluster_num
        video_vector_map[str(line)] = video_vector

    Parallel(n_jobs=16)(delayed(create)(i, line) for i, line in enumerate(lines))

    # for i, line in enumerate(lines):
    #     create(i, line)

    output_file = "/project/hsr/sshalini/LSMA/11775-hws/hw2_code/kmeans/features"
    pickle.dump(video_vector_map, open(output_file, 'wb'))

    print("K-means features generated successfully!")
