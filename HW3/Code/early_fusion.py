import pickle
import numpy as np
import sys

print("Early fusion started")

features_map = {"MFCC" : pickle.load(open("features/MFCC/features", "rb")), 
				"SURF" : pickle.load(open("features/SURF/feature_vectors_bck.pickle", "rb")), 
				"CNN" : pickle.load(open("features/CNN/features_512", "rb")),
				"ASR" : pickle.load(open("features/ASR/features", "rb")),
				"Resnet" : pickle.load(open('features/CNN_resnet/features', 'rb')),
				"Soundnet" : pickle.load(open('features/Soundnet/features', 'rb')),
				"Places" : pickle.load(open('features/Places/features', 'rb')),
				"None" : {}
			} 

# MFCC 400, ASR 5609, SURF 450, CNN 512

f1 = sys.argv[1]
f2 = sys.argv[2]
f3 = sys.argv[3]
f4 = sys.argv[4]
output_file = sys.argv[5]

all_videos = open('list/all.video', 'r')

mfcc_new_map = {}
for entry in features_map["MFCC"].items():
	mfcc_new_map[entry[0].strip()] = entry[1]

features_map["MFCC"] = mfcc_new_map

new_map = {}

for i, video in enumerate(all_videos):
	# print(i)
	video_name = str(video).strip()
	features_list = []
	for f in [f1, f2, f3, f4]:
		if video_name in features_map[f]:
			features_list.append(np.array(features_map[f][video_name]).reshape((1, -1)))
	if len(features_list) == 0:
		continue
	if len(features_list) > 1:
		feature_new = np.concatenate(tuple(features_list), axis=1).flatten()
	else:
		feature_new = features_list[0].flatten()
	if i == 0:
		print(feature_new.shape)
	new_map[video_name] = feature_new

pickle.dump(new_map, open(output_file, 'wb'))

print("Early fusion done")