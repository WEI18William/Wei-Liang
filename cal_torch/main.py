import numpy as np
import nibabel as nib
import os
from center import find_centers
from distance import calculate_distances
from scatter_chart import plot_scatter_3d
from distance_excel import save_distance_to_excel

class_labels = [1, 2, 3, 4, 5]
class_names = ['hemorrhage', 'Cerebellum', 'Brainstem', 'Basal ganglia', 'Dorsal thalamus']
path = '/data/disk3/liangwei2/predict_stack_sort_list/3d_stack_output/'
outpath = '/data/disk3/liangwei2/distance/revised_one/'
list = os.listdir(path)

for i in list:
    path1 = os.path.join(path, i)
    img = nib.load(path1)
    print(i)
    data = np.asanyarray(img.dataobj)
    center = find_centers(data,class_labels,class_names)
    dist = calculate_distances(center,class_names)
    ii = str(i)
    series_png = ii.replace('.nii.gz', '.png')
    series_xlsx = ii.replace('.nii.gz', '.xlsx')
    plot_scatter_3d(center, class_names, class_labels,os.path.join(outpath,'scatter_chart',series_png))
    save_distance_to_excel(dist, class_names, os.path.join(outpath,'distance_chart', series_xlsx))



