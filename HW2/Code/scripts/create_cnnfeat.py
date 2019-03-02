#!/bin/python
import numpy as np
import os
import pickle
from sklearn.cluster.k_means_ import KMeans
import sys

# python create_cnnfeat.py list/all.video

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: {0} vocab_file, file_list".format(sys.argv[0]))
		print("vocab_file -- path to the vocabulary file")
		print("file_list -- the list of videos")
		exit(1)

	all_video_file = str(sys.argv[1])
	all_videos = open(all_video_file, 'r')

	video_vector_map = {}

	cluster_num = 512

	for video in all_videos:
		video_name = str(video).strip().split('.')[0]
		cnn_path = "/project/hsr/sshalini/LSMA/11775-hws/hw2_code/cnn_cropped/" + video_name + ".cnn"
		# cnn_path = "/data/VOL4/ashwinsr/11775-hws/docker/hw2/11775-hws/hw2_code/cnnLatest/" + video_name + ".cnn"
		if os.path.exists(cnn_path) == True:
			feature_vector = pickle.load(open(cnn_path, 'rb'))
			feature_vector = np.average(feature_vector, axis=0)
			feature_vector = np.array(feature_vector) * 1.0 / np.sum(feature_vector)
		else:
			feature_vector = [1.0 / cluster_num] * cluster_num
		video_vector_map[video_name] = feature_vector

	output_file = "/project/hsr/sshalini/LSMA/11775-hws/hw2_code/cnnfeat/features_512"
	pickle.dump(video_vector_map, open(output_file, 'wb'))

	print("CNN features generated successfully!")
