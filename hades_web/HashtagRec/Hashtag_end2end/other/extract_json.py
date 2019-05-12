import os, json
import numpy as np
import sys

min_num = 0
max_num = 0
cnt = 0
tag_len = 0
image_cnt = 0
avgtag_total = 0

path = sys.argv[1]
all_tag_path = path+'/50tags.txt' #'./50hashtags01/50tags.txt'
file_write = open(all_tag_path, 'a')
for file in os.listdir(path):
	if 'rawfeed' not in file and file != '50tags.txt' and file != '50tags_lan.txt' and file != '50tags_emoji.txt':
		tmp_file = path +'/'+ file
		try:
			with open(tmp_file, 'r') as json_file:
				json_data = json.load(json_file)
		except Exception as e:
			print('exception in json file: ', tmp_file)
			raise e
		#os.mkdir()
		
		for post in json_data['posts']:
			if post['tags']:
				tags_list = post['tags']
				sentence = []
				for tag in tags_list:
					sentence.append(tag.split("#")[1] + " ")
				image_cnt += 1
				file_write.writelines(sentence)
				file_write.write('\n')

				tag_len += len(tags_list)
				if cnt == 0:
					min_num = len(tags_list)
				elif len(tags_list) < min_num:
					min_num = len(tags_list)
				if len(tags_list) > max_num:
					max_num = len(tags_list)
	cnt += 1
	
file_write.close()

avgtag_total = tag_len / cnt
print('tag_len', tag_len)
print('avgtag_total: ', avgtag_total)
print('image_cnt: ', image_cnt)
print('min_num: ', min_num)
print('max_num: ', max_num)