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
'''
line_30--进行预测，详细参数看：https://docs.ultralytics.com/modes/predict/#working-with-results
line_40--进行推理可视化保存
line_70--保存mask切割图片
'''

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
                    results = model.predict(source_path, conf=0.5, save_txt=False, save_crop=False, boxes=False, device='0')

                    #预测可视化图片并保存
                    annotated = results[0].plot()
                    # cv2.imwrite(f"test_pics/test3/{a}.jpg", annotated)

                    #获取mask
                    for result in results:
                        masks = result.masks  # Masks object for segmentation masks outputs
                    coordinates = masks.xy

                    b = 1
                    h, w, _ = images.shape
                    black_img = np.zeros([h, w], dtype=np.uint8)
                    res = np.zeros_like(images)
                    for i in coordinates:
                        #mask位置

                        mask = polygons_to_mask2([h, w], i)
                        mask = mask.astype(np.uint8)
                        #显示黑白mask
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

                        #提取>0部分到新图并进行裁剪
                        res[mask > 0] = images[mask > 0]

                        #裁剪后的图
                        masked = res[y1:y2, x1:x2]
                        # cv2.imwrite(f"{target_folder}4_4_{c}.jpg", masked)
                        c+=1

                        polygons = np.asarray(i, np.int32)
                        cv2.fillConvexPoly(black_img, polygons, 1)
                        # plt.subplot(111)
                        # plt.imshow(black_img, 'gray')
                        # plt.show()

                    cv2.imwrite(f"{target_folder}{a}.jpg", black_img*255)
                    cv2.imwrite(f"{target_folder}{a}_ori.jpg", res)

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

if __name__=="__main__":
    source_folder = "./assets/"#
    target_folder = "./detect_bad/"
    model = YOLO("models/best_new.pt")

    save_file(source_folder,target_folder)

