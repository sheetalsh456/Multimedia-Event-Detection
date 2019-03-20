#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
from sklearn import preprocessing
from sklearn.metrics.pairwise import chi2_kernel
import pickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

# python test_svm.py cnn_model.pickle cnnfeat/ 256 op1.txt op2.txt op3.txt

if __name__ == '__main__':
    print("Test svm started")
    # if len(sys.argv) != 5:
    #     print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
    #     print "model_file -- path of the trained svm file"
    #     print "feat_dir -- dir of feature files"
    #     print "feat_dim -- dim of features; provided just for debugging"
    #     print "output_file -- path to save the prediction score"
    #     exit(1)

    # num_clusters = 5606 # for asr
    # num_clusters = 400 # for kmeans
    # num_clusters=10896
    # num_clusters=5462
    # feat_dim = 5609
    model_file = sys.argv[1]
    features_file = sys.argv[2]
    output_file1 = sys.argv[3]
    output_file2 = sys.argv[4]
    output_file3 = sys.argv[5]
    output_file4 = sys.argv[6]

    svm_model = pickle.load(open(model_file,"rb"))
    video_vector_map = pickle.load(open(features_file,"rb"))
    train_videos_file = open('/project/hsr/sshalini/LSMA/11775-hws/hw2_code/list/all_trn.lst', 'r')
    val_videos_file = open('/project/hsr/sshalini/LSMA/11775-hws/hw2_code/list/all_val.lst', 'r')
    test_videos_file = open('/project/hsr/sshalini/LSMA/11775-hws/hw2_code/list/test.video', 'r')

    for key in video_vector_map.items():
        feat_dim = len(key[1])
        break

    num_clusters = feat_dim

    
    # trainX = pickle.load(open(model_file.split('.')[0] + '_trainX.pickle', 'rb'))

    # fwrite1 = open('Files/Test/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file1, 'w')
    # fwrite2 = open('Files/Test/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file2, 'w')
    # fwrite3 = open('Files/Test/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file3, 'w')
    # fwrite4 = open('Files/Test/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file4, 'w')

    # fwrite1 = open('Files/TrainVal_Final/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file1, 'w')
    # fwrite2 = open('Files/TrainVal_Final/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file2, 'w')
    # fwrite3 = open('Files/TrainVal_Final/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file3, 'w')
    # fwrite4 = open('Files/TrainVal_Final/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file4, 'w')

    fwrite1 = open('Files/Test_Final/Early/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file1, 'w')
    fwrite2 = open('Files/Test_Final/Early/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file2, 'w')
    fwrite3 = open('Files/Test_Final/Early/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file3, 'w')
    fwrite4 = open('Files/Test_Final/Early/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file4, 'w')

    # fwrite1 = open('Files/Train/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file1, 'w')
    # fwrite2 = open('Files/Train/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file2, 'w')
    # fwrite3 = open('Files/Train/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file3, 'w')
    # fwrite4 = open('Files/Train/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file4, 'w')

    # fwrite1 = open('Files/Val/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file1, 'w')
    # fwrite2 = open('Files/Val/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file2, 'w')
    # fwrite3 = open('Files/Val/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file3, 'w')
    # fwrite4 = open('Files/Val/' + model_file.split('.')[0].split('/')[-1] + '+' + output_file4, 'w')

    # fwrite1 = open(output_file1, 'w')
    # fwrite2 = open(output_file2, 'w')
    # fwrite3 = open(output_file3, 'w')
    # fwrite4 = open(output_file4, 'w')

    # file_y = open('labels.txt', 'w')

    X = []
    Y = []

    cnt=0
    total=0    

    # for line in train_videos_file:
    #     video_name = str(line).strip().split(' ')[0]
    #     label = str(line).strip().split(' ')[1]
    #     total += 1
    #     if label == "NULL" and cnt < total * 1.0 / 4:
    #         label = 0
    #         cnt += 1
    #     elif label == "NULL":
    #         continue
    #     if video_name in video_vector_map:
    #         video_vector = video_vector_map[video_name]
    #         X.append(video_vector)
    #     else:
    #         X.append([1.0/num_clusters]*num_clusters) # change if needed
    #     Y.append(label)

    # cnt=0
    # total=0

    # for line in val_videos_file:
    #     video_name = str(line).strip().split(' ')[0]
    #     label = str(line).strip().split(' ')[1]
    #     total += 1
    #     if label == "NULL" and cnt < total * 1.0 / 4:
    #         label = 0
    #         cnt += 1
    #     elif label == "NULL":
    #         continue
    #     if video_name in video_vector_map:
    #         video_vector = video_vector_map[video_name]
    #         X.append(video_vector)
    #     else:
    #         X.append([1.0/num_clusters]*num_clusters) # change if needed
    #     Y.append(label)

    for line in test_videos_file:
        video_name = str(line).strip().split(' ')[0]
        if video_name in video_vector_map:
            video_vector = video_vector_map[video_name]
            X.append(video_vector)
        else:
            X.append([1.0/num_clusters]*num_clusters) # change if needed

    # print(len(X), len(X[0]))

    # probs = svm_model.predict(X)
    # print(probs)
    # print(len(probs))
    # exit(0)

    # X = pca_model.transform(X)

    # K = chi2_kernel(X, trainX)

    # print(svm_model.classes_)

    probs = svm_model.predict_proba(X)
    probs1 = probs[:,0] # NULL
    probs2 = probs[:,1]
    probs3 = probs[:,2]
    probs4 = probs[:,3]
    # max_prob = numpy.max(true_probs)
    # min_prob = numpy.min(true_probs)
    # probs_norm = (true_probs - min_prob)*1.0 / (max_prob - min_prob)
    
    for prob_norm1 in probs1:
        fwrite1.write(str(prob_norm1)+'\n')

    for prob_norm2 in probs2:
        fwrite2.write(str(prob_norm2)+'\n')

    for prob_norm3 in probs3:
        fwrite3.write(str(prob_norm3)+'\n')

    for prob_norm4 in probs4:
        fwrite4.write(str(prob_norm4)+'\n')

    # for y in Y:
    #     file_y.write(str(y) + '\n')

    print("test svm done")






