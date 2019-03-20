#!/bin/bash

map_path=/data/VOL4/sshalini/data/speech-kitchen.org/sdalmia/mAP
export PATH=$map_path:$PATH

for feature in "MFCC" "SURF" "CNN" "Resnet" "Soundnet" "Places"
do
	echo "$feature"
	echo "-----------------------------------------" >> "output1.txt"
	echo "$feature" >> "output1.txt"
	python early_fusion.py $feature "None" "None" "None" "EarlyFusion/$feature"
	python train_svm_chi2.py "EarlyFusion/$feature" "model.pickle"
	python test_svm_chi2.py "model.pickle" "EarlyFusion/$feature" "op0.txt" "op1.txt" "op2.txt" "op3.txt"
	ap "list/NULL_val_label" "op0.txt" >> "output1.txt"
	ap "list/P001_val_label" "op1.txt" >> "output1.txt"
	ap "list/P002_val_label" "op2.txt" >> "output1.txt"
	ap "list/P003_val_label" "op3.txt" >> "output1.txt"
	echo "\n" >> "output1.txt"
done

for feature1 in "MFCC" "SURF" "CNN" "Resnet" "Soundnet" "Places"
do
	for feature2 in "MFCC" "SURF" "CNN" "Resnet" "Soundnet" "Places"
	do
		if [ $feature1 != $feature2 ]
		then
			echo "$feature1+$feature2"
			echo "-----------------------------------------" >> "output2.txt"
			echo "$feature1+$feature2" >> "output2.txt"
			python early_fusion.py $feature1 $feature2 "None" "None" "EarlyFusion/$feature1+$feature2"
			python train_svm_chi2.py "EarlyFusion/$feature1+$feature2" "model.pickle"
			python test_svm_chi2.py "model.pickle" "EarlyFusion/$feature1+$feature2" "op0.txt" "op1.txt" "op2.txt" "op3.txt"
			ap "list/NULL_val_label" "op0.txt" >> "output2.txt"
			ap "list/P001_val_label" "op1.txt" >> "output2.txt"
			ap "list/P002_val_label" "op2.txt" >> "output2.txt"
			ap "list/P003_val_label" "op3.txt" >> "output2.txt"
			echo "\n" >> "output2.txt"
		fi
	done
done

for feature1 in "MFCC" "SURF" "CNN" "Resnet" "Soundnet" "Places"
do
	for feature2 in "MFCC" "SURF" "CNN" "Resnet" "Soundnet" "Places"
	do
		if [ $feature1 != $feature2 ]
		then
			for feature3 in "MFCC" "SURF" "CNN" "Resnet" "Soundnet" "Places"
			do
				if [ $feature3 != $feature1 ] && [ $feature3 != $feature2 ]
				then
					echo "$feature1+$feature2+$feature3"
					echo "-----------------------------------------" >> "output3.txt"
					echo "$feature1+$feature2+$feature3" >> "output3.txt"
					python early_fusion.py $feature1 $feature2 $feature3 "None" "EarlyFusion/$feature1+$feature2+$feature3"
					python train_svm_chi2.py "EarlyFusion/$feature1+$feature2+$feature3" "model.pickle"
					python test_svm_chi2.py "model.pickle" "EarlyFusion/$feature1+$feature2+$feature3" "op0.txt" "op1.txt" "op2.txt" "op3.txt"
					ap "list/NULL_val_label" "op0.txt" >> "output3.txt"
					ap "list/P001_val_label" "op1.txt" >> "output3.txt"
					ap "list/P002_val_label" "op2.txt" >> "output3.txt"
					ap "list/P003_val_label" "op3.txt" >> "output3.txt"
					echo "\n" >> "output3.txt"
				fi
			done
		fi
	done
done

for feature1 in "MFCC" "SURF" "CNN" "Resnet" "Soundnet" "Places"
do
	for feature2 in "MFCC" "SURF" "CNN" "Resnet" "Soundnet" "Places"
	do
		if [ $feature1 != $feature2 ]
		then
			for feature3 in "MFCC" "SURF" "CNN" "Resnet" "Soundnet" "Places"
			do
				if [ $feature3 != $feature1 ] && [ $feature3 != $feature2 ]
				then
					for feature4 in "MFCC" "SURF" "CNN" "Resnet" "Soundnet" "Places"
					do
						if [ $feature4 != $feature1 ] && [ $feature4 != $feature2 ] && [ $feature4 != $feature3 ]
						then
							echo "$feature1+$feature2+$feature3+$feature4"
							echo "-----------------------------------------" >> "output4.txt"
							echo "$feature1+$feature2+$feature3+$feature4" >> "output4.txt"
							python early_fusion.py $feature1 $feature2 $feature3 $feature4 "EarlyFusion/$feature1+$feature2+$feature3+$feature4"
							python train_svm_chi2.py "EarlyFusion/$feature1+$feature2+$feature3+$feature4" "model.pickle"
							python test_svm_chi2.py "model.pickle" "EarlyFusion/$feature1+$feature2+$feature3+$feature4" "op0.txt" "op1.txt" "op2.txt" "op3.txt"
							ap "list/NULL_val_label" "op0.txt" >> "output4.txt"
							ap "list/P001_val_label" "op1.txt" >> "output4.txt"
							ap "list/P002_val_label" "op2.txt" >> "output4.txt"
							ap "list/P003_val_label" "op3.txt" >> "output4.txt"
							echo "\n" >> "output4.txt"
						fi
					done
				fi
			done
		fi
	done
done
