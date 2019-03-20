#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
from sklearn.ensemble import BaggingClassifier
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics.pairwise import chi2_kernel
import sys

# python train_svm.py cnnfeat/ cnn_model.pickle 256

# Performs K-means clustering and save the model to a local file

print("Hello")

if __name__ == '__main__':
    print("Train svm started")
    # print(len(sys.argv))
    # if len(sys.argv) != 5:
    #     print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
    #     print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
    #     print "feat_dir -- dir of feature files"
    #     print "feat_dim -- dim of features"
    #     print "output_file -- path to save the svm model"
    #     exit(1)

    # event_name = sys.argv[1] # label
    features_file = sys.argv[1] # cnnfeat
    output_file = sys.argv[2]   

    # feat_dim = 10869
    # feat_dim = 5609
    # feat_dim = 400

    video_vector_map = pickle.load(open(features_file,"rb"))
    train_videos_file = open('/project/hsr/sshalini/LSMA/11775-hws/hw2_code/list/all_trn.lst', 'r')
    val_videos_file = open('/project/hsr/sshalini/LSMA/11775-hws/hw2_code/list/all_val.lst', 'r')

    for key in video_vector_map.items():
        feat_dim = len(key[1])
        print(feat_dim)
        break

    # video_vector_map_new = {}
    # for entry in video_vector_map.items():
    #     video_vector_map_new[entry[0].strip()] = entry[1]

    # can change to make 1/vocab_size vectors for videos not in video_vector_map

    # X = []
    # Y = []
    # for line in train_videos_file:
    #     video_name = str(line).split(' ')[0]
    #     label = str(line).split(' ')[1].split('\n')[0]
    #     if label != "NULL":
    #         if label == event_name:
    #             label = 1
    #         else:
    #             label = 0
    #         if video_name in video_vector_map_new:
    #             video_vector = video_vector_map_new[video_name]
    #             X.append(video_vector)
    #         else:
    #             X.append([1.0 / feat_dim] * feat_dim)
    #         Y.append(label)

    # for line in val_videos_file:
    #     video_name = str(line).split(' ')[0]
    #     label = str(line).split(' ')[1].split('\n')[0]
    #     if label != "NULL":
    #         if label == event_name:
    #             label = 1
    #         else:
    #             label = 0
    #         if video_name in video_vector_map_new:
    #             video_vector = video_vector_map_new[video_name]
    #             X.append(video_vector)
    #         else:
    #             X.append([1.0 / feat_dim] * feat_dim)
    #         Y.append(label)

    total = 0
    cnt = 0

    X = []
    Y = []
    for line in train_videos_file:
        video_name = str(str(line).split(" ")[0])
        label = str(line).split(" ")[1].split('\n')[0]
        total += 1
        if label == "NULL" and cnt < total * 1.0 / 4:
            label = 0
            cnt += 1
        elif label == "NULL":
            continue
        if label == "P001":
            label = 1
        elif label == "P002":
            label = 2
        elif label == "P003":
            label = 3
        if video_name in video_vector_map:
            video_vector = video_vector_map[video_name]
            X.append(video_vector)
        else:
            X.append([1.0 / feat_dim] * feat_dim)
        Y.append(label)

    cnt = 0
    total = 0

    for line in val_videos_file:
        video_name = str(str(line).split(" ")[0])
        label = str(line).split(" ")[1].split('\n')[0]
        total += 1
        if label == "NULL" and cnt < total * 1.0 / 4:
            label = 0
            cnt += 1
        elif label == "NULL":
            continue
        if label == "P001":
            label = 1
        elif label == "P002":
            label = 2
        elif label == "P003":
            label = 3
        if video_name in video_vector_map:
            video_vector = video_vector_map[video_name]
            X.append(video_vector)
        else:
            X.append([1.0 / feat_dim] * feat_dim)
        Y.append(label)


    # def normalize_sklearn(input):
    #     tfidf = TfidfTransformer(norm='l2', use_idf=False, sublinear_tf=False)
    #     mat = tfidf.fit_transform(input).toarray()
    #     return mat

    # X = normalize_sklearn(np.array(X))

    # print(np.array(X).shape, np.array(Y).shape)

    # X = np.array(X)
    # x_sum = np.sum(X, axis=1)
    # x_sum[x_sum == 0] = 1
    # X = X * 1.0 / x_sum.reshape((-1,1))


    # x = np.copy(X)
    # x[x > 0] = 1
    # idf = np.log((X.shape[0] + 1) * (1.0 / (np.sum(x, axis=0) + 1)))

    # tfidf = TfidfTransformer(norm='l2', use_idf=True, sublinear_tf=False, smooth_idf = False)
    # mat = tfidf.fit_transform(X).toarray()
# 
    # clf = SVC(gamma='auto', probability = True)
    # clf = SVC(kernel='poly') # 0.17, 0.18, 0.20
    # clf = SVC(kernel='sigmoid') # 0.377, 0.18, 0.2
    # clf = SVC(kernel='linear') # 0.377, 0.18, 0.2
    # clf = SVC(kernel='rbf') # 0.377, 0.18, 0.2
    # clf.fit(X, Y)
# 
    # clf = SVC(kernel=chi2_kernel, probability=True)
    # clf = MLPClassifier(max_iter=600, hidden_layer_sizes=(512, 512, 512, 512, 512, 256, 256))

    # best
    # K = chi2_kernel(X, gamma=1.5)
    # clf = MLPClassifier(hidden_layer_sizes=(2048, 2048, 1024), max_iter=600) # best

    # clf = MLPClassifier(hidden_layer_sizes=(1024, 512), max_iter=600) # best

    # clf = MLPClassifier()

    clf = MLPClassifier(hidden_layer_sizes=(1024, 512))

    # clf = MLPClassifier(max_iter=600) # best


    # scaler = MinMaxScaler(feature_range=[0, 1])
    # X = scaler.fit_transform(X)

    #Fitting the PCA algorithm with our Data
    # pca = PCA(n_components=900).fit(X)
    # X = pca.transform(X)
    #Plotting the Cumulative Summation of the Explained Variance
    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of Components')
    # plt.ylabel('Variance (%)') #for each component
    # plt.title('Pulsar Dataset Explained Variance')
    # plt.savefig("plot.png")
    # exit(0)

    # clf = KNeighborsClassifier(n_neighbors=20)

    # clf = AdaBoostClassifier(n_estimators=20)

    # clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, max_depth=2, random_state=0)

    # clf = GaussianNB()

    # clf = DecisionTreeClassifier(random_state=0, max_depth=6)

    # clf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0)

    clf.fit(X, Y) 

    pickle.dump(clf, open(output_file, 'wb'))
    # cPickle.dump(pca, open("pca_model", 'w+'))

    # pickle.dump(X, open(output_file.split('.')[0] + '_trainX.pickle', 'wb'))


    print('SVM trained successfully')

    print("Train svm done")
