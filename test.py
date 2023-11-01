import os
import shutil
import numpy as np

# # 定义源文件夹和目标文件夹
# src_folder = './datasets/bookshelf-point/images/train/'
# dst_folder = './datasets/bookshelf-point/images/val/'
#
# # 定义源标签文件夹和目标标签文件夹
# src_label_folder = './datasets/bookshelf-point/labels/train/'
# dst_label_folder = './datasets/bookshelf-point/labels/val/'
#
# # 获取源文件夹中的所有文件
# all_files = os.listdir(src_folder)
#
# # 随机选择20%的文件作为验证集
# val_files = np.random.choice(all_files, size=int(0.2 * len(all_files)), replace=False)
#
# # 将选中的文件和对应的标签移动到目标文件夹
# for file in val_files:
#     # 移动图片
#     shutil.move(os.path.join(src_folder, file), os.path.join(dst_folder, file))
#
#     # 移动对应的标签
#     label_file = file[:-4] + '.txt'  # 假设标签文件是.txt格式，并且和图片文件名相同
#     shutil.move(os.path.join(src_label_folder, label_file), os.path.join(dst_label_folder, label_file))

def copy_images(src_dir, dst_dir):
    for dirpath, dirnames, filenames in os.walk(src_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                src_file = os.path.join(dirpath, filename)
                shutil.copy(src_file, dst_dir)

# 移动文件
src_dir = './Original'
dst_dir = './total'

copy_images(src_dir, dst_dir)