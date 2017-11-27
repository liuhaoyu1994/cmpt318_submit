import pandas as pd
import numpy as np
import cv2
import glob
import sys
import re

class MyImage:
	def __init__(self, img_name):
		self.img = cv2.imread(img_name)
		self.__name = img_name

	def __str__(self):
		return self.__name
#https://stackoverflow.com/questions/44663347/python-opencv-reading-the-image-file-name

datetime_re = re.compile(r'-(\d\d\d\d)(\d\d)(\d\d)(\d\d)(\d\d)')

def get_datetime(txt):
    match = datetime_re.search(txt)
    if match:
        datetime = match.group(1)+'-'+match.group(2)+'-'+match.group(3)+' '+match.group(4)+':'+match.group(5)
        return datetime
    else:
        return None

def path_to_time(filename):
    datetime = get_datetime(filename)
    return datetime

def images_to_csv(in_dir,out_dir):
	for filename in glob.glob(in_dir+'/'+'*.jpg'):
		image = MyImage(filename)
		BGR = image.img
		height, width, channels = BGR.shape
		b,g,r = cv2.split(BGR)
		RGB = np.dstack((r,g,b))
		RGB=RGB.flatten()
		datetime = tuple(((path_to_time(str(image))),))
		content = tuple(RGB.tolist())
		file = open(out_dir+'/'+path_to_time(str(image))+'.csv','w')
		file.write(str(content)[1:-1])
		file.close()
		#output.to_csv(out_dir+'/'+path_to_time(str(image))+'.csv')

def main():
	in_dir = sys.argv[1]
	out_dir = sys.argv[2]
	images_to_csv(in_dir,out_dir)

if __name__ == '__main__':
	main()