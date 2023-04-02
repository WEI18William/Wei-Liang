import pandas as pd
import os

# 设置目录路径和输出文件名
directory = '/data/disk3/liangwei2/distance/revised_one/distance_chart/'
output_filename = os.path.join(directory,'average_distance.xlsx')

# 创建空的数据帧来存储所有表格的数据
all_data = pd.DataFrame()

# 循环遍历目录中的所有Excel文件
for filename in os.listdir(directory):
    if filename.endswith('.xlsx'):  # 只处理扩展名为.xlsx的文件
        filepath = os.path.join(directory, filename)
        # 读取Excel文件并将数据添加到all_data
        data = pd.read_excel(filepath, index_col=0)
        all_data = all_data.add(data, fill_value=0)

# 计算所有表格的平均值
mean_data = all_data / len(os.listdir(directory))

# 将结果写入新的Excel文件
mean_data.to_excel(output_filename)
