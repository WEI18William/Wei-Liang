import glob

import PIL.Image as Image
import cv2
import os

path = "/home/liangwei/two_dimension/"
outpath = "/home/liangwei/transition_segment/label"

label_path = os.path.join(path,'label')
label_dir = glob.glob(os.path.join(label_path,'*',"*"))
for file in label_dir:
    file_str = file.split('/')
    file_seris = file_str[5]
    file_case = file_str[6]
    file_num = file_case.split('.')[0]
    mask = Image.open(file).convert('RGB')
    dir_path = os.path.join(outpath, file_seris)
    if os.path.isdir(dir_path):
        pass
    else:
        os.mkdir(dir_path)
    mask.save(os.path.join(dir_path, file_num+'.png'))