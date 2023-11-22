from flask import Flask, request, jsonify
import torch
from PIL import Image
import base64
import io
import numpy as np
import cv2
from ultralytics import YOLO
import pickle
import hashlib
import time
import os
from scipy.ndimage import gaussian_filter
import asyncio


app = Flask(__name__)
import json
# 模型加载

model = "models/best_point.pt"
model = YOLO(model)

#unique signal
def generate_unique_id():
    timestamp = str(int(time.time() * 1000)) # 时间戳
    uid = hashlib.md5(timestamp.encode()).hexdigest() # 使用MD5哈希函数生成唯一标识符
    return uid

# 保存推理结果

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
def predict(image,parament):
    img = preprocess(image)

    img = np.asarray(img)

    h, w, _ = img.shape
    #dir = make_file(img)
    print(parament)
    #inference
    results = model(img, conf=float(parament['conf']), imgsz=640, save_txt=False, save_crop=False, boxes=False, device='0')
    keypoints_ = results[0].keypoints
    if keypoints_ is not None:
        keypoints = keypoints_.xyn.cpu().numpy()
        dict1 = {}
        for points in keypoints:
            # points[:, 0] = points[:, 0] * w
            # points[:, 1] = points[:, 1] * h
            #
            # mask = polygons_to_mask2([h, w], points)
            #
            # # 使用边界框截取矩形区域
            # res[mask > 0] = img[mask > 0]
            #
            # x = points[:, 0]
            # y = points[:, 1]
            # y1 = int(min(y))
            # y2 = int(max(y))
            # x1 = int(min(x))
            # x2 = int(max(x))
            #
            # crop_img = res[y1:y2, x1:x2]
            #
            # dict1[(x1, y1)] = crop_img
            #
            # # 按照键（即左上角坐标）对字典进行排序
            # sorted_items = sorted(dict1.items())
            #
            # # 从排序后的列表中提取图像，并将它们添加到新的列表中
            # list1 = [item[1] for item in sorted_items]

            points[:, 0] = points[:, 0] * w
            points[:, 1] = points[:, 1] * h

            x = points[:, 0]
            y = points[:, 1]
            y1 = int(min(y))
            x1 = int(min(x))

            dict1[(x1, y1)] = points.tolist()

            # 按照键（即左上角坐标）对字典进行排序
            sorted_items = sorted(dict1.items())

            # 从排序后的列表中提取图像，并将它们添加到新的列表中
            list1 = [item[1] for item in sorted_items]
    else:
        list1 = 'NONE'
    #save_results(list1, dir)
    return list1

# 预处理
def preprocess(image):
    return Image.open(io.BytesIO(image))

# 返回base64
def transform(outputs):
    point_json = json.dumps(outputs)

    return point_json

def make_file(img):
    unique_id = str(generate_unique_id())
    subdir = os.path.join('app_results', unique_id)
    os.makedirs(subdir, exist_ok=True)
    filename = f'{int(time.time() * 1000)}.jpg'
    file_path = os.path.join(subdir, filename)
    cv2.imwrite(file_path, img)

    return subdir

def save_results(results, input_dir):
    unique_id = str(generate_unique_id())
    subdir = os.path.join(input_dir, unique_id)
    os.makedirs(subdir, exist_ok=True)

    for i, result in enumerate(results):
        filename = f'{i}_{int(time.time() * 1000)}.jpg'
        file_path = os.path.join(subdir, filename)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        cv2.imwrite(file_path, result)
    # 文件名为推理结果的索引加上时间戳后缀

#接口
@app.route('/predict', methods=['POST'])
def get_prediction():
    file = request.files['file']
    parament = request.files['parament'].read()
    parament = pickle.loads(parament)
    img_bytes = file.read()
    result = predict(img_bytes, parament)
    result = transform(result)

    return jsonify({'content': result})

# host = os.environ.get('APP_HOST')
# port = os.environ.get('APP_PORT')
host = "0.0.0.0"
port = "5000"

app.run(host=host, port=port, debug=False)
