from ultralytics import YOLO
import os
import cv2
import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
# from labelme import utils
import numpy as np
import glob
import PIL.Image
import PIL.ImageDraw
import os,sys
from pycocotools import coco
import numpy as np
from PIL import Image

source_folder = "./test_pics/test6/4_4/"#
target_folder = "./test_pics/pic/test1"
model = YOLO("models/best_new.pt")
test = "./test_pics/test4/1/"

# 遍历文件夹中的所有文件
def save_file(source_folder, target_folder):
        # 检查文件是否为图像文件
        c = 1
        for file_name in os.listdir(source_folder):
                a = file_name.split('.')[0]
                if file_name.endswith(".jpeg") or file_name.endswith(".jpg") or file_name.endswith(".png"):
                    # 使用OpenCV读取图像
                    source_path = os.path.join(source_folder, file_name)
                    images = cv2.imread(source_path)

                    #进行预测
                    results = model.predict(source_path, conf=0.7, save_txt=False, save_crop=False, boxes=False, device='0')

                    #预测可视化图片并保存
                    annotated = results[0].plot()
                    cv2.imwrite(f"test_pics/test3/{a}.jpg", annotated)

                    #获取mask
                    for result in results:
                        masks = result.masks  # Masks object for segmentation masks outputs
                    coordinates = masks.xy
                    b = 1
                    for i in coordinates:
                        #mask位置
                        h, w, _ = images.shape
                        mask = polygons_to_mask2([h, w], i)
                        mask = mask.astype(np.uint8)
                        # 显示黑白mask
                        # plt.subplot(111)
                        # plt.imshow(mask, 'gray')
                        # plt.show()

                        # mask所在坐标矩形框
                        x = i[:, 0]
                        y = i[:, 1]
                        y1 = int(min(y))
                        y2 = int(max(y))
                        x1 = int(min(x))
                        x2 = int(max(x))
                        # 创建与原图大小全黑图，用于提取.
                        res = np.zeros_like(images)
                        #提取>0部分到新图并进行裁剪
                        res[mask > 0] = images[mask > 0]
                        #裁剪后的图
                        masked = res[y1:y2, x1:x2]
                        if os.path.exists(test):
                            pass
                        else:
                            os.makedirs("./test_pics/test4/1/")
                        cv2.imwrite(f"./test_pics/test6/4_4/book/4_4_{c}.jpg", masked)
                        # cv2.imwrite(f"{source_folder}/book/1/1_2_{c}.jpg", masked)
                        c+=1
                        #重新提取坐标为ploygon格式（坐标都存在一个列表中）
                        # d = []  # polygon
                        # coodrs = []
                        # for j in i:
                        #     coodrs.append(int(j[0]))
                        #     coodrs.append(int(j[1]))
                        # c.append(coodrs)
                        # d.append(i[:, 0])
                        # d.append(i[:, 1])


                        # #显示mask
                        # plt.subplots(111)
                        # plt.imshow(roi, 'gray')

                        # cv2.imwrite(f"test_pics/test3/{a+1}.jpg", masked)


#返回True or False Bool类型
def polygons_to_mask(img_shape, polygons):
    '''
    边界点生成mask
    :param img_shape: [h,w]
    :param polygons: labelme JSON中的边界点格式 [[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]
    :return:
    '''
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def polygons_to_mask2(img_shape, polygons):
    '''
    边界点生成mask
    :param img_shape: [h,w]
    :param polygons: labelme JSON中的边界点格式 [[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]
    :return:
    '''
    mask = np.zeros(img_shape, dtype=np.uint8)
    polygons = np.asarray(polygons, np.int32) # 这里必须是int32，其他类型使用fillPoly会报错
    # cv2.fillPoly(mask, polygons, 1) # 非int32 会报错
    cv2.fillConvexPoly(mask, polygons, 1)  # 非int32 会报错
    return mask
def mask2box(mask):
    '''从mask反算出其边框
    mask：[h,w]  0、1组成的图片
    1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
    '''
    # np.where(mask==1)
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    # 解析左上角行列号
    left_top_r = np.min(rows)  # y
    left_top_c = np.min(clos)  # x

    # 解析右下角行列号
    right_bottom_r = np.max(rows)
    right_bottom_c = np.max(clos)

    return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2] 对应Pascal VOC 2007 的bbox格式
    # return [left_top_c, left_top_r, right_bottom_c - left_top_c,
    #         right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

save_file(source_folder,target_folder)
# for image in os.listdir(image_dir):
#     for image in
