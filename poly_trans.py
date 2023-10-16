import json
import cv2
import os
import numpy as np
'''
三个参数分别为：json文件，原图图片，保存图片地址
'''
def ploy_trans(txt_dir,img_dir,img_save):
    if not os.path.exists(img_save):
        os.makedirs(img_save)
    a = 1
    for file_name in os.listdir(txt_dir):
        if file_name.endswith('.json'):
            json_file = f"{file_name.split('.')[0]}.jpg"
            json_path = os.path.join(txt_dir, file_name)
            for img_name in os.listdir(img_dir):
                if json_file == img_name and img_name.endswith('.jpg'):
                    img_num = img_name.split('.')[0]
                    #读取照片
                    img_pth = os.path.join(img_dir, img_name)
                    img = cv2.imread(img_pth)
                    #json读取
                    with open(json_path) as f:
                        txt = json.load(f)
                    for point in txt['shapes']:
                        points = point['points']
                        points = np.int32(points)
                        #提取矩形坐标
                        mask = np.zeros_like(img)

                        # 将四点构成的区域填充为白色
                        cv2.fillConvexPoly(mask, np.array(points), (255, 255, 255))

                        # 使用掩码裁剪
                        cropped = cv2.bitwise_and(img, mask)

                        x1 = int(min(p[0] for p in points))
                        y1 = int(min(p[1] for p in points))
                        x2 = int(max(p[0] for p in points))
                        y2 = int(max(p[1] for p in points))
                        #裁剪后区域
                        cropped1 = cropped[y1:y2, x1:x2]

                        #保存
                        cv2.imwrite(f"{img_save}/4_4_{a}.jpg", cropped1)
                        a+=1

if __name__=="__main__":
    txt_dir = "test_pics/test6/4_5/"
    img_dir = "test_pics/test6/4_5/"
    img_save = "test_pics/test6/4_5/book"

    ploy_trans(txt_dir, img_dir, img_save)



