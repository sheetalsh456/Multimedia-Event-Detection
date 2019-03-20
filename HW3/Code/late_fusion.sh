#!/bin/bash

map_path=/data/VOL4/sshalini/data/speech-kitchen.org/sdalmia/mAP
export PATH=$map_path:$PATH


for feature in "MFCC" "SURF" "Resnet" "Soundnet" "Places"
do
	echo "$feature"
	echo "$feature" >> "LateFusion/Extra/output1.txt"
	python late_fusion.py $feature "None" "None" "None" "None"
	python late_fusion_test.py $feature "None" "None" "None" "None"
	ap "list/NULL_val_label" "LateFusion/Files/Val/$feature+op0.txt" >> "LateFusion/Extra/output1.txt"
	ap "list/P001_val_label" "LateFusion/Files/Val/$feature+op1.txt" >> "LateFusion/Extra/output1.txt"
	ap "list/P002_val_label" "LateFusion/Files/Val/$feature+op2.txt" >> "LateFusion/Extra/output1.txt"
	ap "list/P003_val_label" "LateFusion/Files/Val/$feature+op3.txt" >> "LateFusion/Extra/output1.txt"
	echo "\n" >> "LateFusion/Extra/output1.txt"
done


for feature1 in "MFCC" "SURF" "Resnet" "Soundnet" "Places"
do
	for feature2 in "MFCC" "SURF" "Resnet" "Soundnet" "Places"
	do
		if [ $feature1 != $feature2 ]
		then
			echo "$feature1+$feature2"
			echo "$feature1+$feature2" >> "LateFusion/Extra/output2.txt"
			python late_fusion.py $feature1 $feature2 "None" "None" "None"
			python late_fusion_test.py $feature1 $feature2 "None" "None" "None"
			ap "list/NULL_val_label" "LateFusion/Files/Val/$feature1+$feature2+op0.txt" >> "LateFusion/Extra/output2.txt"
			ap "list/P001_val_label" "LateFusion/Files/Val/$feature1+$feature2+op1.txt" >> "LateFusion/Extra/output2.txt"
			ap "list/P002_val_label" "LateFusion/Files/Val/$feature1+$feature2+op2.txt" >> "LateFusion/Extra/output2.txt"
			ap "list/P003_val_label" "LateFusion/Files/Val/$feature1+$feature2+op3.txt" >> "LateFusion/Extra/output2.txt"
			echo "\n" >> "LateFusion/Extra/output2.txt"
		fi
	done
done


for feature1 in "MFCC" "SURF" "Resnet" "Soundnet" "Places"
do
	for feature2 in "MFCC" "SURF" "Resnet" "Soundnet" "Places"
	do
		if [ $feature1 != $feature2 ]
		then
			for feature3 in "MFCC" "SURF" "Resnet" "Soundnet" "Places"
			do
				if [ $feature3 != $feature1 ] && [ $feature3 != $feature2 ]
				then
					echo "$feature1+$feature2+$feature3"
					echo "$feature1+$feature2+$feature3" >> "LateFusion/Extra/output3.txt"
					python late_fusion.py $feature1 $feature2 $feature3 "None" "None"
					python late_fusion_test.py $feature1 $feature2 $feature3 "None" "None"
					ap "list/NULL_val_label" "LateFusion/Files/Val/$feature1+$feature2+$feature3+op0.txt" >> "LateFusion/Extra/output3.txt"
					ap "list/P001_val_label" "LateFusion/Files/Val/$feature1+$feature2+$feature3+op1.txt" >> "LateFusion/Extra/output3.txt"
					ap "list/P002_val_label" "LateFusion/Files/Val/$feature1+$feature2+$feature3+op2.txt" >> "LateFusion/Extra/output3.txt"
					ap "list/P003_val_label" "LateFusion/Files/Val/$feature1+$feature2+$feature3+op3.txt" >> "LateFusion/Extra/output3.txt"
					echo "\n" >> "LateFusion/Extra/output3.txt"
				fi
			done
		fi
	done
done


for feature1 in "MFCC" "SURF" "Resnet" "Soundnet" "Places"
do
	for feature2 in "MFCC" "SURF" "Resnet" "Soundnet" "Places"
	do
		if [ $feature1 != $feature2 ]
		then
			for feature3 in "MFCC" "SURF" "Resnet" "Soundnet" "Places"
			do
				if [ $feature3 != $feature1 ] && [ $feature3 != $feature2 ]
				then
					for feature4 in "MFCC" "SURF" "Resnet" "Soundnet" "Places"
					do
						if [ $feature4 != $feature1 ] && [ $feature4 != $feature2 ] && [ $feature4 != $feature3 ]
						then
							echo "$feature1+$feature2+$feature3+$feature4"
							echo "$feature1+$feature2+$feature3+$feature4" >> "LateFusion/Extra/output4.txt"
							python late_fusion.py $feature1 $feature2 $feature3 $feature4 "None"
							python late_fusion_test.py $feature1 $feature2 $feature3 $feature4 "None"
							ap "list/NULL_val_label" "LateFusion/Files/Val/$feature1+$feature2+$feature3+$feature4+op0.txt" >> "LateFusion/Extra/output4.txt"
							ap "list/P001_val_label" "LateFusion/Files/Val/$feature1+$feature2+$feature3+$feature4+op1.txt" >> "LateFusion/Extra/output4.txt"
							ap "list/P002_val_label" "LateFusion/Files/Val/$feature1+$feature2+$feature3+$feature4+op2.txt" >> "LateFusion/Extra/output4.txt"
							ap "list/P003_val_label" "LateFusion/Files/Val/$feature1+$feature2+$feature3+$feature4+op3.txt" >> "LateFusion/Extra/output4.txt"
							echo "\n" >> "LateFusion/Extra/output4.txt"
						fi
					done
				fi
			done
		fi
	done
done

