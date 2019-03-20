from os import listdir
import numpy as np
import pickle
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.ensemble import ExtraTreesClassifier

# print("Late fusion started")

# file1 = sys.argv[1]
# file2 = sys.argv[2]
# file3 = sys.argv[3]
# file4 = sys.argv[4]
# file5 = sys.argv[5]

# files = []

# if file2 == "None":
# 	files.append('Train/' + file1)
# 	model_name = file1
# elif file3 == "None":
# 	files.extend(['Train/' + file1, 'Train/' + file2])
# 	model_name = file1 + '+' + file2
# elif file4 == "None":
# 	files.extend(['Train/' + file1, 'Train/' + file2, 'Train/' + file3])
# 	model_name = file1 + '+' + file2 + '+' + file3
# elif file5 == "None":
# 	files.extend(['Train/' + file1, 'Train/' + file2, 'Train/' + file3, 'Train/' + file4])
# 	model_name = file1 + '+' + file2 + '+' + file3 + '+' + file4
# else:
# 	files.extend(['Train/' + file1, 'Train/' + file2, 'Train/' + file3, 'Train/' + file4, 'Train/' + file5])
# 	model_name = file1 + '+' + file2 + '+' + file3 + '+' + file4 + '+' + file5

files = ['TrainVal_Final/Places', 'TrainVal_Final/Resnet', 'TrainVal_Final/MFCC']

model_name = 'Places+Resnet+MFCC'

features = []

for i, file in enumerate(files):
	lines0 = open('Files/' + file.split('.')[0] + '+op0.txt', 'r').readlines()
	lines1 = open('Files/' + file.split('.')[0] + '+op1.txt', 'r').readlines()
	lines2 = open('Files/' + file.split('.')[0] + '+op2.txt', 'r').readlines()
	lines3 = open('Files/' + file.split('.')[0] + '+op3.txt', 'r').readlines()
	features.extend([lines0, lines1, lines2, lines3])

X = np.array(features).T.astype(np.float64)

Y = open('labels.txt', 'r').readlines()


# clf = MLPClassifier(hidden_layer_sizes=(1024, 512), max_iter=600) # best

clf = MLPClassifier()

# clf = ExtraTreesClassifier(n_estimators=15)

clf.fit(X, Y) 
pickle.dump(clf, open('LateFusion/Test/' + model_name + '.pickle', 'wb'))

print("Late fusion done")
