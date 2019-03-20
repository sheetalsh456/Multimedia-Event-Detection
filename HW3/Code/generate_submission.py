import numpy as np
import csv
import os
import sys

print("generate submission started")

filename1 = sys.argv[1]
filename2 = sys.argv[2]
filename3 = sys.argv[3]
filename4 = sys.argv[4]

output_file = sys.argv[5]

file1 = open(filename1, 'r')
file2 = open(filename2, 'r')
file3 = open(filename3, 'r')
file4 = open(filename4, 'r')

csv_file = open(output_file, "w+")
writer = csv.writer(csv_file)

test_videos_file = open("/project/hsr/sshalini/LSMA/11775-hws/hw2_code/list/test.video")

writer.writerow(["VideoID", "Label"])
for line1, line2, line3, line4, test_line in zip(file1, file2, file3, file4, test_videos_file):
	values = [float(line1.split()[0]), float(line2.split()[0]), float(line3.split()[0]), float(line4.split()[0])]
	writer.writerow([str(test_line.split()[0]), np.argmax(values)])

print("generate submission done\n\n")
