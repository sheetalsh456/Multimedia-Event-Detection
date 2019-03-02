#!/bin/python
import numpy as np
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import nltk
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import sys

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print "Usage: {0} vocab_file, file_list".format(sys.argv[0])
		print "vocab_file -- path to the vocabulary file"
		print "file_list -- the list of videos"
		exit(1)

	all_video_file = str(sys.argv[2])
	asr_vocab_list_file = str(sys.argv[1])

	asr_vocab_list = cPickle.load(open(asr_vocab_list_file,"rb"))

	all_videos = open(all_video_file, 'r')
	all_videos_map = {}

	final_list = []

	video_vector_map = {}

	for video in all_videos:
		video_name = str(video).strip()
		if video_name not in all_videos_map:
			all_videos_map[video_name] = 1

	tokenizer = RegexpTokenizer(r'\w+')
	ps = PorterStemmer()

	print(len(asr_vocab_list))

	for video_name in all_videos_map:
		video_name = str(video_name).split('.')[0]
		asr_path = "/project/hsr/sshalini/LSMA/11775-hws/hw1_code/asr/" + video_name + ".txt"
		if os.path.exists(asr_path) == True:
			line = open(asr_path, 'r')
			# data = tokenizer.tokenize((str(line.read()).lower().strip()))
			data = str(line.read()).strip().split()
			word_map = {}
			for word in data:
				# word = ps.stem(word)
				if word not in word_map:
					word_map[word] = 0
				word_map[word] += 1
			asr_feature_vector = []
			for asr_word in asr_vocab_list:
				if asr_word not in word_map:
					asr_feature_vector.append(0)
				else:
					asr_feature_vector.append(word_map[asr_word])
			if np.sum(asr_feature_vector) == 0:
				asr_feature_vector = [1.0 / len(asr_vocab_list)] * len(asr_vocab_list)
			else:
				asr_feature_vector = np.array(asr_feature_vector) * 1.0 / np.sum(asr_feature_vector)
		else:
			asr_feature_vector = [1.0 / len(asr_vocab_list)] * len(asr_vocab_list)
		video_vector_map[video_name] = asr_feature_vector

	# for file in os.listdir("/project/hsr/sshalini/LSMA/11775-hws/hw1_code/asr/"):
	# 	if file.endswith(".txt") and str(file).split('.')[0] in all_videos_map:
	# 		line = open("/project/hsr/sshalini/LSMA/11775-hws/hw1_code/asr/" + file, 'r')
	# 		data = str(line.read()).strip().split()
	# 		word_map = {}
	# 		for word in data:
	# 			if word not in word_map:
	# 				word_map[word] = 0
	# 			word_map[word] += 1
	# 		asr_feature_vector = []
	# 		for asr_word in asr_vocab_list:
	# 			if asr_word not in word_map:
	# 				asr_feature_vector.append(0)
	# 			else:
	# 				asr_feature_vector.append(word_map[asr_word])
	# 		if np.sum(asr_feature_vector) == 0:
	# 			asr_feature_vector = [1.0 / len(asr_vocab_list)] * len(asr_vocab_list)
	# 		else:
	# 			asr_feature_vector /= np.sum(asr_feature_vector)
	# 		video_vector_map[str(file).split('.')[0]] = asr_feature_vector

	output_file = "/project/hsr/sshalini/LSMA/11775-hws/hw1_code/asrfeat/features"
	cPickle.dump(video_vector_map, open(output_file, 'w+'))



	print "ASR features generated successfully!"
