#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
from sklearn import preprocessing
from sklearn.metrics.pairwise import chi2_kernel
import cPickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
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
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file1 = sys.argv[4]
    output_file2 = sys.argv[5]
    output_file3 = sys.argv[6]

    num_clusters = feat_dim



    video_vector_map = cPickle.load(open(feat_dir+'features',"rb"))
    val_videos_file = open('/project/hsr/sshalini/LSMA/11775-hws/hw1_code/list/test.video', 'rb')

    video_vector_map_new = {}
    for entry in video_vector_map.iteritems():
        video_vector_map_new[entry[0].strip()] = entry[1]

    svm_model = cPickle.load(open(model_file,"rb"))
    # pca_model = cPickle.load(open("pca_model","rb"))

    fwrite1 = open(output_file1, 'w')
    fwrite2 = open(output_file2, 'w')
    fwrite3 = open(output_file3, 'w')

    X = []
    for line in val_videos_file:
        video_name = str(line).strip()
        if video_name in video_vector_map_new:
            video_vector = video_vector_map_new[video_name]
            X.append(video_vector)
        else:
            X.append([1.0/num_clusters]*num_clusters) # change if needed

    # print(len(X), len(X[0]))

    # probs = svm_model.predict(X)
    # print(probs)
    # print(len(probs))
    # exit(0)

    # X = pca_model.transform(X)

    probs = svm_model.predict_proba(X)
    probs1 = probs[:,0]
    probs2 = probs[:,1]
    probs3 = probs[:,2]
    # max_prob = numpy.max(true_probs)
    # min_prob = numpy.min(true_probs)
    # probs_norm = (true_probs - min_prob)*1.0 / (max_prob - min_prob)
    
    for prob_norm1 in probs1:
        fwrite1.write(str(prob_norm1)+'\n')

    for prob_norm2 in probs2:
        fwrite2.write(str(prob_norm2)+'\n')

    for prob_norm3 in probs3:
        fwrite3.write(str(prob_norm3)+'\n')






