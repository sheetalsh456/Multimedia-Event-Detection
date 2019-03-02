#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import collections
import nltk
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import operator
import sys

if __name__ == '__main__':

	asr_vocab_map_file = "asr_vocab_map"
	asr_vocab_list_file = "asr_vocab_list"
	all_video_file = str(sys.argv[1])

	vocab = {}

	all_videos = open(all_video_file, 'r')
	all_videos_map = {}

	for video in all_videos:
		video_name = str(video).strip()
		if video_name not in all_videos_map:
			all_videos_map[video_name] = 1

	tokenizer = RegexpTokenizer(r'\w+')
	ps = PorterStemmer()
	# wordnet_lemmatizer = WordNetLemmatizer()

	num_files = 0
	for file in os.listdir("/project/hsr/sshalini/LSMA/11775-hws/hw1_code/asr/"):
		if file.endswith(".txt") and str(file).split('.')[0] in all_videos_map:
			num_files += 1
			line = open("/project/hsr/sshalini/LSMA/11775-hws/hw1_code/asr/" + file, 'r')
			# data = tokenizer.tokenize((str(line.read()).lower().strip()))
			data = str(line.read()).strip().split()
			# print(data)
			for word in data:
				# word = ps.stem(word)
				if word not in vocab:
					vocab[word] = 0
				vocab[word] += 1


	# cPickle.dump(vocab, open(asr_vocab_map_file, 'w'))
	num_words = 0

	vocab_list = []
	sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
	for key in sorted_vocab:
		if key[1] >= 2 and key[1] <= 800:
			num_words += 1
			vocab_list.append(key[0]) # 5606 terms
	print(num_words)

	# exit(0)

	cPickle.dump(vocab_list, open(asr_vocab_list_file, 'w'))

	print "ASR vocab created successfully!"
