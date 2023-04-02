import nibabel as nib
import numpy as np
import os

def check_value(path):
    # 定义要读取的NIfTI文件路径

    # 加载NIfTI文件
    img = nib.load(path)

    # 获取图像数据
    data = img.get_fdata()

    # 获取大于0的数值
    positive_values = np.unique(data[data > 0])
    # 打印大于0的数值
    print(positive_values)

path = '/data/disk3/liangwei2/predict_stack_sort_list/3d_stack_output/'
outpath = '/data/disk3/distance_2_out/'
list = os.listdir(path)
for i in list:
    path1 = os.path.join(path,i)
    check_value(path1)
