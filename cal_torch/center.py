import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass
import os

def find_centers(data,class_labels,class_names):
    # Define class labels

    # Create dictionary to store centers for each class
    centers_dict = {}
    for i, label in enumerate(class_labels):
        centers_dict[i] = []

    # Loop through class labels and find centers for each connected component
    for i, label in enumerate(class_labels):
        # Create binary mask for current class
        mask = (data == label)
        # Label connected components in the mask
        labels, num_components = ndimage.label(mask)
        # Loop through connected components and find their centers
        for j in range(1, num_components+1):
            # Create binary mask for current component
            component_mask = (labels == j)
            # Find center of mass for current component
            center = center_of_mass(component_mask)
            # Add center to centers list for current class
            centers_dict[i].append(center)



    # Print centers_list for each class
    for i, label in enumerate(class_labels):
        centers = centers_dict[i]
        print(f"Class {label} ({class_names[i]}): {len(centers)} centers found")
        for j, center in enumerate(centers):
            print(f"    Center {j + 1}: {center}")


    return centers_dict

if __name__ == '__main__':

    path = '/data/disk3/liangwei2/predict_stack_sort_list/3d_stack_output/'
    outpath = '/data/disk3/distance_2_out/'
    list = os.listdir(path)
    for i in list:
        path1 = os.path.join(path,i)
        img = nib.load(path1)
        print(i)
        data = np.asanyarray(img.dataobj)
        find_centers(data)
