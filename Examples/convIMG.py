import os
import cv2
root = os.getcwd()
path = root+'/BEND/'
batch = [".".join(f.split(".")[:-1]) for f in os.listdir(path)]
for sample in batch:
	img_path = path+sample
	img = cv2.imread(img_path+'.bmp')
	cv2.imwrite(sample + '.jpg', img)
print("done!")
