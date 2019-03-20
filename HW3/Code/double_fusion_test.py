from os import listdir
import numpy as np
import pickle
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import chi2_kernel
import sys


# print("Late fusion test started")

# file1 = sys.argv[1]
# file2 = sys.argv[2]
# file3 = sys.argv[3]
# file4 = sys.argv[4]
# file5 = sys.argv[5]

# files = []

# if file2 == "None":
# 	files.append('Val/' + file1)
# 	model_name = file1
# elif file3 == "None":
# 	files.extend(['Val/' + file1, 'Val/' + file2])
# 	model_name = file1 + '+' + file2
# elif file4 == "None":
# 	files.extend(['Val/' + file1, 'Val/' + file2, 'Val/' + file3])
# 	model_name = file1 + '+' + file2 + '+' + file3
# elif file5 == "None":
# 	files.extend(['Val/' + file1, 'Val/' + file2, 'Val/' + file3, 'Val/' + file4])
# 	model_name = file1 + '+' + file2 + '+' + file3 + '+' + file4
# else:
# 	files.extend(['Val/' + file1, 'Val/' + file2, 'Val/' + file3, 'Val/' + file4, 'Val/' + file5])
# 	model_name = file1 + '+' + file2 + '+' + file3 + '+' + file4 + '+' + file5

files = ['Test_Final/Resnet+Soundnet', 'Test_Final/Places+MFCC']

model_name = 'Resnet+Soundnet_Places+MFCC'

features = []

for i, file in enumerate(files):
	lines0 = open('Files/' + file.split('.')[0] + '+op0.txt', 'r').readlines()
	lines1 = open('Files/' + file.split('.')[0] + '+op1.txt', 'r').readlines()
	lines2 = open('Files/' + file.split('.')[0] + '+op2.txt', 'r').readlines()
	lines3 = open('Files/' + file.split('.')[0] + '+op3.txt', 'r').readlines()
	features.extend([lines0, lines1, lines2, lines3])

X = np.array(features).T.astype(np.float64)

print(X.shape)

model_file = 'DoubleFusion/Test/' + model_name + '.pickle'

svm_model = pickle.load(open(model_file, "rb"))

probs = svm_model.predict_proba(X)
probs1 = probs[:,0] # NULL
probs2 = probs[:,1]
probs3 = probs[:,2]
probs4 = probs[:,3]

fwrite1 = open('DoubleFusion/Files/Test/' + model_file.split('.')[0].split('/')[-1] + '+op0.txt', 'w')
fwrite2 = open('DoubleFusion/Files/Test/' + model_file.split('.')[0].split('/')[-1] + '+op1.txt', 'w')
fwrite3 = open('DoubleFusion/Files/Test/' + model_file.split('.')[0].split('/')[-1] + '+op2.txt', 'w')
fwrite4 = open('DoubleFusion/Files/Test/' + model_file.split('.')[0].split('/')[-1] + '+op3.txt', 'w')

for prob_norm1 in probs1:
    fwrite1.write(str(prob_norm1)+'\n')

for prob_norm2 in probs2:
    fwrite2.write(str(prob_norm2)+'\n')

for prob_norm3 in probs3:
    fwrite3.write(str(prob_norm3)+'\n')

for prob_norm4 in probs4:
    fwrite4.write(str(prob_norm4)+'\n')

print("Late fusion test done")
    
