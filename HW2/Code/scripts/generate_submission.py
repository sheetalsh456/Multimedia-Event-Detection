import numpy as np
import csv
import os
import sys

filename1 = sys.argv[1]
filename2 = sys.argv[2]
filename3 = sys.argv[3]

# filename1 = "/project/hsr/sshalini/LSMA/11775-hws/hw1_code/final/gradientboosting_test1.lst"
# filename2 = "/project/hsr/sshalini/LSMA/11775-hws/hw1_code/final/svm_test2.lst"
# filename3 = "/project/hsr/sshalini/LSMA/11775-hws/hw1_code/final/mlp_test3.lst"

file1 = open(filename1, 'r')
file2 = open(filename2, 'r')
file3 = open(filename3, 'r')

csv_file = open("submission_final1.csv", "w+")
writer = csv.writer(csv_file)

test_videos_file = open("/project/hsr/sshalini/LSMA/11775-hws/hw2_code/list/test.video")

writer.writerow(["VideoID", "Label"])
for line1, line2, line3, test_line in zip(file1, file2, file3, test_videos_file):
	values = [float(line1.split()[0]), float(line2.split()[0]), float(line3.split()[0])]
	writer.writerow([str(test_line.split()[0]), np.argmax(values)+1])
