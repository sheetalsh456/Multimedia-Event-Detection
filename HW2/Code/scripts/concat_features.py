import pickle
import numpy as np

cnn_feat_map = pickle.load(open('/project/hsr/sshalini/LSMA/11775-hws/hw2_code/cnnfeat/features_512', 'rb'))
surf_feat_map = pickle.load(open('/project/hsr/sshalini/LSMA/11775-hws/hw2_code/kmeans/feature_vectors_bck.pickle', 'rb'))

all_videos = open('list/all.video', 'r')

new_map = {}
num = 0

for video in all_videos:
	num += 1
	video_name = str(video).strip()
	cnn_changed = cnn_feat_map[video_name].reshape((1, -1))
	surf_changed = np.array(surf_feat_map[video_name]).reshape((1, -1))
	feature_new = np.concatenate((cnn_changed, surf_changed), axis=1).flatten()
	new_map[video_name] = feature_new

output_file = "/project/hsr/sshalini/LSMA/11775-hws/hw2_code/concatfeat/features"
pickle.dump(new_map, open(output_file, 'wb'))

print(num)