#!/usr/bin/env python3

import torch
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import os
import sys
import threading
import numpy as np
import yaml
import cv2
import pickle
import pdb
from joblib import Parallel, delayed

alexnet = models.alexnet(pretrained=True)
modules=list(alexnet.children())[:-1]
alexnet=nn.Sequential(*modules)
for p in alexnet.parameters():
    p.requires_grad = False

def get_cnn_features_from_video(i, line_new, downsampled_videos_new, cnn_features_folderpath_new, keyframe_interval_new, hessian_threshold_new):
    # TODO

    video_name = line_new.replace('\n', '')
    downsampled_video_filename = os.path.join(downsampled_videos_new, video_name + '.ds.mp4')
    cnn_feat_video_filename = os.path.join(cnn_features_folderpath_new, video_name + '.cnn')

    if not os.path.isfile(downsampled_video_filename):
        return

    # downsampled_video_filename = "/project/hsr/sshalini/LSMA/11775-hws/hw2_code/downsampled_videos/HVC5995.ds.mp4"
    imgs = []

    for img in get_keyframes(downsampled_video_filename, keyframe_interval_new):
        img = np.array(img) # (120, 160, 3)
        img = np.moveaxis(img, -1, 0) # (3, 120, 160)
        imgs.append(img)

    # print("images appending done")

    imgs = torch.Tensor(np.array(imgs))

    img_var = Variable(imgs) # assign it to a variable
    try:
        features_var = alexnet(img_var) # get the output from the last hidden layer of the pretrained resnet
    except:
        print("returned")
        return
    features = features_var.data # get the tensor out of the variable
    
    print(i, downsampled_video_filename, features.shape)

    # pooled_features = np.average(features, axis=(2, 3))
    pooled_features = np.dstack(features)
    pooled_features = np.vstack(pooled_features)
    pooled_features = np.moveaxis(pooled_features, -1, 0)

    print(pooled_features.shape)

    cnn_filename = downsampled_video_filename.split('/')[-1].split('.')[0]
    pickle.dump(pooled_features, open("cnn_cropped/" + cnn_filename + ".cnn", 'wb'))

def get_keyframes(downsampled_video_filename, keyframe_interval):

    # Create video capture object
    # print("inside get keyframes")
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    while True:
        # print("inside while true")
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield img
    video_cap.release()


# python cnn_feat_extraction.py list/all.video config_10.yaml 

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))

    # Get parameters from config file
    keyframe_interval = my_params.get('keyframe_interval')
    hessian_threshold = my_params.get('hessian_threshold')
    cnn_features_folderpath = my_params.get('cnn_features')
    downsampled_videos = my_params.get('downsampled_videos_new')

    # TODO: Create SURF object

    # Check if folder for SURF features exists
    if not os.path.exists(cnn_features_folderpath):
        os.mkdir(cnn_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    fread = open(all_video_names, "r")
    lines = fread.readlines()

    Parallel(n_jobs=6)(delayed(get_cnn_features_from_video)(i, line, downsampled_videos, cnn_features_folderpath, keyframe_interval, hessian_threshold) for i, line in enumerate(lines))