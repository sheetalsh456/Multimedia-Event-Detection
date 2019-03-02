from subprocess import call
from joblib import Parallel, delayed
import os

#Python2

def process_data(pid, file_list):
	old_filename = "/data/VOL4/sshalini/data/speech-kitchen.org/sdalmia/11775_videos/video/"+file_list+".mp4"
	new_filename = "/project/hsr/sshalini/LSMA/11775-hws/hw2_code/downsampled_videos_new/"+file_list+".ds.mp4"
	print("\n --------------------------------------------- " + str(pid+1) + " ----------------------------------------\n")
	print(old_filename)
	print(new_filename)
	try:
		with open(old_filename,'r') as ex:
			pass
	except:
		print(old_filename)
		print(new_filename)
		return
	os_string = "ffmpeg -y -ss 0 -i "+old_filename+" -an -vf scale=160x120 -strict experimental -t 60 -r 15 "+new_filename
	os.system(os_string)

chunks = []
with open("/project/hsr/sshalini/LSMA/11775-hws/hw2_code/list/all.video","r") as ifile:
	for line in ifile:
		chunks.append(line.strip())
	print(len(chunks))

Parallel(n_jobs = 1)(delayed(process_data)(i, chunk) for i, chunk in enumerate(chunks))

# for i, chunk in enumerate(chunks):
# 	print("\n --------------------------------------------- " + str(i+1) + " ----------------------------------------\n")
# 	process_data(i, chunk)
