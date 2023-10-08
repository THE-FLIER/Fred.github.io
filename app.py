from flask import Flask, request, jsonify
import torch
import yaml
from PIL import Image
import base64
import io
import numpy as np
import cv2
from ultralytics import YOLO
app = Flask(__name__)
import json
# 模型加载

model = "models/best_new.pt"
model = YOLO(model)

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

# 预测
def predict(image):
    # 预处理图片
    img = preprocess(image)
    with torch.no_grad():
        out = model.predict(img, conf=0.7, save_txt=False, save_crop=False, boxes=False, device='cpu')
        for result in out:
            masks = result.masks  # Masks object for segmentation masks outputs
        coordinates = masks.xy
        list1 = []
        for i in coordinates:
            # mask位置
            h, w = result.orig_shape
            mask = polygons_to_mask2([h, w], i)
            mask = mask.astype(np.uint8)

            # mask所在坐标矩形框
            x = i[:, 0]
            y = i[:, 1]
            y1 = int(min(y))
            y2 = int(max(y))
            x1 = int(min(x))
            x2 = int(max(x))
            # 创建与原图大小全黑图，用于提取.
            img = np.asarray(img)
            res = np.zeros_like(img)
            # 提取>0部分到新图并进行裁剪
            res[mask > 0] = img[mask > 0]
            # 裁剪后的图
            masked = res[y1:y2, x1:x2]
            list1.append(masked)
        return list1


# 预处理
def preprocess(image):

    return Image.open(io.BytesIO(image))

# Base64转PIL Image
def base64_to_pil(image):
    img = Image.open(io.BytesIO(base64.b64decode(image)))
    return img

def transform(outputs):
    image_list = []
    for img in outputs:
        # 转换为Image格式
        pil_img = Image.fromarray(img)
        # 编码为base64
        buff = io.BytesIO()
        pil_img.save(buff, format="jpg")
        img_str = base64.b64encode(buff.getvalue()).decode('utf-8')
        # 添加到列表
        image_list.append(img_str)
    return image_list

#接口
@app.route('/predict', methods=['POST'])
def get_prediction():
    file = request.files['file']
    img_bytes = file.read()
    result = predict(img_bytes)
    result = transform(result)
    return jsonify({'content': result})

if __name__ == '__main__':
    app.run()
