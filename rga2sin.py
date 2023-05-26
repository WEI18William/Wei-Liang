import glob

from tqdm import tqdm
import numpy as np
import cv2
import os

# 标签中每个RGB颜色的值
VOC_COLORMAP = np.array([[0, 0, 0], [255, 0, 255], [255, 255, 0], [255, 0, 0],
                [0, 128, 0],[0,128,128]])
# 标签其标注的类别
VOC_CLASSES = ['background', 'hemorrhage', 'yellow', 'red', 'green',
               'blue']

path = '/data/disk3/liangwei/SUMChosp100plus2023/unet/data/transition/labels/*/*'
outpath = '/data/disk3/liangwei/SUMChosp100plus2023/unet/data/transition/labels/'

label = glob.glob(path)
for file in label:
    file_str = file.split('/')
    file_seris = file_str[9]
    print(file_seris)
    file_case = file_str[10]
    print(file_case)
    file_num = file_case.split('.')[0]
    mask = cv2.imread(file)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) # 通道转换
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    # 标签处理
    dir_path = os.path.join(outpath, file_seris)

    if os.path.isdir(dir_path):
        pass
    else:
        os.mkdir(dir_path)

    for ii, label in enumerate(VOC_COLORMAP):
        locations = np.all(mask == label, axis=-1)
        label_mask[locations] = ii
    # 标签保存
    cv2.imwrite(os.path.join(dir_path,file_case), label_mask)
