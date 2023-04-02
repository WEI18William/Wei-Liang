import os
import cv2
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
import sys
sys.setrecursionlimit(20000)


def extract_arrays(file_name):
    img = nib.load(file_name)
    data = img.get_fdata()

    arrays = [np.where(data == i, 1, 0) for i in range(1, 6)]

    return arrays


def dilate_volume(array, iterations=1):
    structuring_element = np.array([
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    ], dtype=bool)

    dilated_array = binary_dilation(array, structure=structuring_element, iterations=iterations)
    return dilated_array


def dfs(matrix, visited, x, y):
    if x < 0 or x >= len(matrix) or y < 0 or y >= len(matrix[0]) or visited[x][y] or matrix[x][y] == 0:
        return []

    visited[x][y] = True
    points = [(x, y)]

    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        points += dfs(matrix, visited, x + dx, y + dy)

    return points


def find_connected_regions(matrix):
    visited = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    connected_regions = []

    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            if not visited[x][y] and matrix[x][y] == 1:
                connected_regions.append(dfs(matrix, visited, x, y))

    return connected_regions


def overlapping_area(region1, region2):
    region1_set = set(region1)
    region2_set = set(region2)

    overlap = region1_set.intersection(region2_set)

    return len(overlap)

def non_zero_area(merged_matrix):
    area = 0
    for row in merged_matrix:
        for element in row:
            if element != 0:
                area += 1
    return area


def merge_matrices(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Both matrices should have the same dimensions.")

    merged_matrix = [[0 for _ in range(len(matrix1[0]))] for _ in range(len(matrix1))]

    for x in range(len(matrix1)):
        for y in range(len(matrix1[0])):
            if matrix1[x][y] == 1 or matrix2[x][y] == 1:
                merged_matrix[x][y] = 1

    return merged_matrix

def detect_in_silece(array1,array2):
    z_dim = len(array1[0][0])
    weight_area = []
    for i in range(z_dim):
        silce1 = array1[:,:,i]
        silce2 = array2[:, :, i]
        connected_regions1 = find_connected_regions(silce1)
        connected_regions2 = find_connected_regions(silce2)

        max_overlap_area = 0
        for region1 in connected_regions1:
            for region2 in connected_regions2:
                overlap_area = overlapping_area(region1, region2)
                max_overlap_area = max(max_overlap_area, overlap_area)

        #print("Maximum overlap area:", max_overlap_area)
        merged_matrix = merge_matrices(silce1,silce2)
        area = non_zero_area(merged_matrix)
        #print("The area of non-zero values in the merged matrix is:", area)
        if area == 0:
            weight_area0 = 0
        else:
            weight_area0 = max_overlap_area/ area
        weight_area.append(weight_area0)
    weight = sum(weight_area)
    return weight

def cosine_similarity(list1, list2):
    # 将列表转换为NumPy数组
    vec1 = np.array(list1)
    vec2 = np.array(list2)

    # 计算向量的点积
    dot_product = np.dot(vec1, vec2)

    # 计算向量的范数（长度）
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # 计算余弦相似度
    cos_sim = dot_product / (norm1 * norm2)
    return cos_sim

def txt(i,weight_list,path):
    with open(os.path.join(path,"output.txt"), "w") as file:
        file.write(i + "\n")
        file.write("_ICH_classification_weight: ")
        for weight in weight_list:
            file.write(str(weight) + " ")




def main():
    fold_gt_path = '/data/disk3/liangwei2/gt_stack_sort_list/gt_3d/'
    fold_pre_path = '/data/disk3/liangwei2/predict_stack_sort_list/3d_stack_output'
    list_gt = os.listdir(fold_gt_path)
    for i in list_gt:
        weight_list0 = []
        nii_file0 = os.path.join(fold_gt_path, i)
        arrays0 = extract_arrays(nii_file0)
        array1_binary0 = arrays0[0].astype(bool)
        other_arrays0 = arrays0[1:]
        other_arrays_binary0 = [arr0.astype(bool) for arr0 in other_arrays0]
        dilated_array1_0 = dilate_volume(array1_binary0, iterations=1)
        for arr0 in other_arrays_binary0:
            weight0 = detect_in_silece(dilated_array1_0, arr0)
            weight_list0.append(weight0)
        print(i,"_ICH_classification_weight:", weight_list0)
        txt(i, weight_list0, '/data/disk3/liangwei2/gt_stack_sort_list')

        weight_list1 = []
        nii_file1 = os.path.join(fold_pre_path, i)
        arrays1 = extract_arrays(nii_file1)
        array1_binary1 = arrays1[0].astype(bool)
        other_arrays1 = arrays1[1:]
        other_arrays_binary1 = [arr1.astype(bool) for arr1 in other_arrays1]
        dilated_array1_1 = dilate_volume(array1_binary1, iterations=1)
        for arr1 in other_arrays_binary1:
            weight1 = detect_in_silece(dilated_array1_1, arr1)
            weight_list1.append(weight1)
        print(i, "_ICH_predict_classification_weight:", weight_list1)
        txt(i,weight_list1,'/data/disk3/liangwei2/predict_stack_sort_list')
        similarity = cosine_similarity(weight_list0,weight_list1)
        print(similarity)
        similarity_value = str(similarity)

        with open("/data/disk3/liangwei2/similarity.txt", "w") as file:
            file.write("Cosine similarity: " + similarity_value)




if __name__ == "__main__":
    main()


