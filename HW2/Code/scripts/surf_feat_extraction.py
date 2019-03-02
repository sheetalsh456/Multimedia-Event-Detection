#!/usr/bin/env python3

import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb
from joblib import Parallel, delayed


def get_surf_features_from_video(i, line_new, downsampled_videos_new, surf_features_folderpath_new, keyframe_interval_new, hessian_threshold_new):
    # TODO

    video_name = line_new.replace('\n', '')
    downsampled_video_filename = os.path.join(downsampled_videos_new, video_name + '.ds.mp4')
    surf_feat_video_filename = os.path.join(surf_features_folderpath_new, video_name + '.surf')

    if not os.path.isfile(downsampled_video_filename):
        return

    surf = cv2.xfeatures2d.SURF_create(hessian_threshold_new)
    # downsampled_video_filename = "/project/hsr/sshalini/LSMA/11775-hws/hw2_code/downsampled_videos/HVC1012.ds.mp4"
    surf.setExtended(1)
    all_keyframes = []

    for frame in get_keyframes(downsampled_video_filename, keyframe_interval_new):
        # print('inside for loop')
        keypoint, descriptor = surf.detectAndCompute(frame,None)
        all_keyframes.append(descriptor)

    print(i, downsampled_video_filename, len(all_keyframes))
    # if len(all_keyframes) > 0:
    #     print(all_keyframes[0].shape)
    surf_filename = downsampled_video_filename.split('/')[-1].split('.')[0]
    pickle.dump(all_keyframes, open("surf/" + surf_filename + ".surf", 'wb'))

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
    surf_features_folderpath = my_params.get('surf_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # TODO: Create SURF object

    # Check if folder for SURF features exists
    if not os.path.exists(surf_features_folderpath):
        os.mkdir(surf_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    fread = open(all_video_names, "r")
    lines = fread.readlines()

    Parallel(n_jobs=16)(delayed(get_surf_features_from_video)(i, line, downsampled_videos, surf_features_folderpath, keyframe_interval, hessian_threshold) for i, line in enumerate(lines))