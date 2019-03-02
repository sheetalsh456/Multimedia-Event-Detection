#!/bin/bash

# This script performs a complete Media Event Detection pipeline (MED) using video features:
# a) preprocessing of videos, b) feature representation,
# c) computation of MAP scores, d) computation of class labels for kaggle submission.

# You can pass arguments to this bash script defining which one of the steps you want to perform.
# This helps you to avoid rewriting the bash script whenever there are
# intermediate steps that you don't want to repeat.

# execute: bash run.pipeline.sh -p true -f true -m true -k true -y filepath

# Reading of all arguments:
while getopts p:f:m:k:y: option		# p:f:m:k:y: is the optstring here
	do
	case "${option}"
	in
	p) PREPROCESSING=${OPTARG};;       # boolean true or false
	f) FEATURE_REPRESENTATION=${OPTARG};;  # boolean
	m) MAP=${OPTARG};;                 # boolean
	k) KAGGLE=$OPTARG;;                # boolean
    y) YAML=$OPTARG;;                  # path to yaml file containing parameters for feature extraction
	esac
	done

#export PATH=~/anaconda3/bin:$PATH
#export PATH="/home/ramons/tools/ffmpeg-3.2.4/build/bin:$PATH"

if [ "$PREPROCESSING" = true ] ; then

    # echo "#####################################"
    # echo "#         PREPROCESSING             #"
    # echo "#####################################"

    # steps only needed once
    video_path=/data/VOL4/sshalini/data/speech-kitchen.org/sdalmia/11775_videos/video  # path to the directory containing all the videos.
    # mkdir -p list downsampled_videos surf cnn kmeans  # create folders to save features
    # awk '{print $1}' ../hw1_code/list/train > list/train.video  # save only video names in one file (keeping first column)
    # awk '{print $1}' ../hw1_code/list/val > list/val.video
    # cat /project/hsr/sshalini/LSMA/11775-hws_old/hw1_code/list/train.video /project/hsr/sshalini/LSMA/11775-hws_old/hw1_code/list/val.video /project/hsr/sshalini/LSMA/11775-hws_old/hw1_code/list/test.video > /project/hsr/sshalini/LSMA/11775-hws_old/hw1_code/list/all.video    #save all video names in one file
    # downsampling_frame_len=60
    # downsampling_frame_rate=15

    # 1. Downsample videos into shorter clips with lower frame rates.
    # TODO: Make this more efficient through multi-threading f.ex.
    # start=`date +%s`
    #for line in $(cat "list/all.video"); do
    #    ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t $downsampling_frame_len -r $downsampling_frame_rate downsampled_videos/$line.ds.mp4
    #done
    # python parallel.py
    # end=`date +%s`
    # runtime=$((end-start))
    # echo "Downsampling took: $runtime" #28417 sec around 8h without parallelization
    # exit 1

    # 2. TODO: Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ...)
    # echo "SURF extraction started"
    # python surf_feat_extraction_old.py -i list/all.video config.yaml
    # python surf_feat_extraction.py /project/hsr/sshalini/LSMA/11775-hws/hw2_code/list/all.video /project/hsr/sshalini/LSMA/11775-hws/hw2_code/config.yaml

    # echo "CNN extraction started"

    # python cnn_feat_extraction.py /project/hsr/sshalini/LSMA/11775-hws/hw2_code/list/all.video /project/hsr/sshalini/LSMA/11775-hws/hw2_code/config.yaml

    # 3. TODO: Extract CNN features from keyframes of downsampled videos

    # python select_frames.py list/train.video list/val.video 0.2 select.surf.csv
	

fi

# exit 1

# if [ "$FEATURE_REPRESENTATION" = true ] ; then

    # echo "#####################################"
    # echo "#  SURF FEATURE REPRESENTATION      #"
    # echo "#####################################"

    # 1. TODO: Train kmeans to obtain clusters for SURF features

    # python train_kmeans.py select.surf.csv 400 kmeans_surf.model 
    # exit 1


    # 2. TODO: Create kmeans representation for SURF features

	# echo "#####################################"
 #    echo "#   CNN FEATURE REPRESENTATION      #"
 #    echo "#####################################"

	# 1. TODO: Train kmeans to obtain clusters for CNN features

    # python create_kmeans.py kmeans_surf.model 400 list/all.video


    # 2. TODO: Create kmeans representation for CNN features

# fi



if [ "$MAP" = true ] ; then

    echo "#######################################"
    echo "# MAP results #"
    echo "#######################################"

    # Paths to different tools;
    # map_path=/home/ubuntu/tools/mAP
    # export PATH=$map_path:$PATH

    map_path=/data/VOL4/sshalini/data/speech-kitchen.org/sdalmia/mAP
    export PATH=$map_path:$PATH

    # MAP

    # python train_svm_chi2.py concatfeat/ model.pickle 962
    # python test_svm_chi2.py model.pickle concatfeat/ 962 op1.txt op2.txt op3.txt
    # ap list/P001_val_label op1.txt
    # ap list/P002_val_label op2.txt
    # ap list/P003_val_label op3.txt

    # Test

    python train_svm_chi2.py concatfeat/ model.pickle 962
    python test_svm_chi2.py model.pickle concatfeat/ 962 op1.txt op2.txt op3.txt
    python generate_submission.py op1.txt op2.txt op3.txt

    # 1. TODO: Train SVM with OVR _susing only videos in training set.
    

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

	# 4. TODO: Test SVM with test set saving scores for submission

    # echo "#######################################"
    # echo "# MED with CNN Features: MAP results  #"
    # echo "#######################################"


    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

	# 4. TODO: Test SVM with test set saving scores for submission

fi


# if [ "$KAGGLE" = true ] ; then

    # echo "##########################################"
    # echo "# MED with SURF Features: KAGGLE results #"
    # echo "##########################################"

    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

    # 4. TODO: Test SVM with test set saving scores for submission


    # echo "##########################################"
    # echo "# MED with CNN Features: KAGGLE results  #"
    # echo "##########################################"

    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

	# 4. TODO: Test SVM with test set saving scores for submission

# fi
