#!/bin/python
# Randomly select 

import numpy
import os
import sys
import pickle

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: {0} file_list select_ratio output_file".format(sys.argv[0]))
        print("file_list -- the list of video names")
        print("select_ratio -- the ratio of frames to be randomly selected from each audio file")
        print("output_file -- path to save the selected frames (feature vectors)")
        exit(1)

    file_list_train = sys.argv[1]
    file_list_val = sys.argv[2]
    output_file = sys.argv[4]
    ratio = float(sys.argv[3])

    fread_train = open(file_list_train,"r")
    fread_val = open(file_list_val,"r")
    fwrite = open(output_file,"w")

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first 
    # num_of_frame * ratio rows
    numpy.random.seed(18877)

    train_cnt_not_stack = 0
    val_cnt_not_stack = 0

    for i, line in enumerate(fread_train.readlines()):
        print("select_frames train -------------------------------------> " + str(i+1))
        print(str(line).replace('\n', ''))
        surf_path = "surf/" + line.replace('\n','') + ".surf"
        if os.path.exists(surf_path) == False:
            continue
        keyframes = pickle.load(open(surf_path, 'rb'))

        try:
            keyframes_new = [frame for frame in keyframes if frame is not None]
            array = numpy.vstack(keyframes_new)
            print("Vstack done!!\n")

        except:
            print("Didn't vstack :(\n")
            train_cnt_not_stack += 1
            continue

        numpy.random.shuffle(array)
        select_size = int(array.shape[0] * ratio)
        feat_dim = array.shape[1]

        for n in range(select_size):
            line = str(array[n][0])
            for m in range(1, feat_dim):
                line += ';' + str(array[n][m])
            fwrite.write(line + '\n')       

    for i, line in enumerate(fread_val.readlines()):
        print("select_frames val -------------------------------------> " + str(i+1))
        print(str(line).replace('\n', ''))
        surf_path = "surf/" + line.replace('\n','') + ".surf"
        if os.path.exists(surf_path) == False:
            continue
        keyframes = pickle.load(open(surf_path, 'rb'))

        try:
            keyframes_new = [frame for frame in keyframes if frame is not None]
            array = numpy.vstack(keyframes_new)
            print("Vstack done!!\n")

        except:
            print("Didn't vstack :(\n")
            val_cnt_not_stack += 1
            continue

        numpy.random.shuffle(array)
        select_size = int(array.shape[0] * ratio)
        feat_dim = array.shape[1]

        for n in range(select_size):
            line = str(array[n][0])
            for m in range(1, feat_dim):
                line += ';' + str(array[n][m])
            fwrite.write(line + '\n')       

    fwrite.close()
    print("select frames done")
    print("Train not stacked : " + str(train_cnt_not_stack))
    print("Val not stacked : " + str(val_cnt_not_stack))

