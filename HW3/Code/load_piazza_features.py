import numpy as np
import pickle
import os

all_videos_file = open('list/all.video')

feature_vectors_map = {}

for i, video in enumerate(all_videos_file.readlines()):
	print(i)
	video_name = str(video).strip()
	soundnet_path = 'features/Piazza/soundnet/' + video_name + '.feats'
	if os.path.exists(soundnet_path) == True:
		print("Exists")
		feature_vector = np.genfromtxt(soundnet_path, delimiter=';')
	else:
		print("Doesnt exist")
		feature_vector = [1.0 / 1401] * 1401
	feature_vectors_map[video_name] = feature_vector

pickle.dump(feature_vectors_map, open('features/Soundnet/features', 'wb'))

for i, video in enumerate(all_videos_file.readlines()):
	print(i)
	video_name = str(video).strip()
	cnn_path = 'features/Piazza/resnet50/' + video_name + '.npy'
	if os.path.exists(cnn_path) == True:
		print("Exists")
		feature_vector = np.load(cnn_path)
	else:
		print("Doesnt exist")
		feature_vector = [1.0 / 2048] * 2048
	feature_vectors_map[video_name] = feature_vector

pickle.dump(feature_vectors_map, open('features/CNN_resnet/features', 'wb'))

for i, video in enumerate(all_videos_file.readlines()):
	print(i)
	video_name = str(video).strip()
	cnn_path = 'features/Piazza/places/' + video_name + '.npy'
	if os.path.exists(cnn_path) == True:
		print("Exists")
		feature_vector = np.load(cnn_path)
	else:
		print("Doesnt exist")
		feature_vector = [1.0 / 4096] * 4096
	feature_vectors_map[video_name] = feature_vector

pickle.dump(feature_vectors_map, open('features/Places/features', 'wb'))


