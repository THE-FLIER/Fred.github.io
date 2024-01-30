import os
import shutil
import numpy as np
import cv2


def dataset_split(src_folder,dst_folder,src_label_folder,dst_label_folder):
    # 获取源文件夹中的所有文件
    all_files = os.listdir(src_folder)

    # 随机选择25%的文件作为验证集
    val_files = np.random.choice(all_files, size=int(0.1 * len(all_files)), replace=False)

    # 将选中的文件和对应的标签移动到目标文件夹
    for file in val_files:
        # 移动图片
        shutil.move(os.path.join(src_folder, file), os.path.join(dst_folder, file))

        # 移动对应的标签
        label_file = file[:-4] + '.txt'  # 假设标签文件是.txt格式，并且和图片文件名相同
        shutil.move(os.path.join(src_label_folder, label_file), os.path.join(dst_label_folder, label_file))

# # 定义源文件夹和目标文件夹
# src_folder = 'dataset/multi_points/eleven_points/images/train'
# dst_folder = 'dataset/multi_points/eleven_points/images/val'
#
# # 定义源标签文件夹和目标标签文件夹
# src_label_folder = 'dataset/multi_points/eleven_points/labels/train'
# dst_label_folder = 'dataset/multi_points/eleven_points/labels/val'

#dataset_split(src_folder,dst_folder,src_label_folder,dst_label_folder)

import os
#重命名
def rename_to_jpg(directory):
    for filename in os.listdir(directory):
        if not filename.endswith(".jpg"):
            base = os.path.splitext(filename)[0]
            os.rename(directory + "/" + filename, directory + "/" + base + ".jpg")

# 使用方法：将下面的"path_to_your_directory"替换为你的文件夹路径
#rename_to_jpg("/home/zf/yolov8/dataset/ori/1000/img")

import os
import shutil

def copy_matching_txt_files(img_directory, txt_directory, new_directory):
    # 创建新文件夹（如果不存在）
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # 获取图片文件夹中所有文件的前缀
    img_prefixes = {os.path.splitext(filename)[0] for filename in os.listdir(img_directory) if filename.endswith(".jpg")}

    # 遍历文本文件夹，复制所有前缀匹配的.txt文件
    for filename in os.listdir(txt_directory):
        if filename.endswith(".jpg") and os.path.splitext(filename)[0] in img_prefixes:
            shutil.copy(os.path.join(txt_directory, filename), new_directory)

# 使用方法：将下面的"path_to_your_img_directory", "path_to_your_txt_directory" 和 "path_to_your_new_directory" 替换为你的文件夹路径
#copy_matching_txt_files("/home/zf/yolov8/dataset/lishui_shelves/images/val", "/home/zf/yolov8/dataset/bookshelf_extract/images/val", "dataset/shelves_test")
