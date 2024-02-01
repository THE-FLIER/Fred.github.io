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
import math
from scipy.ndimage import gaussian_filter
import asyncio

app = Flask(__name__)
import json
# 模型加载

model = "models/best.pt"
model = YOLO(model)

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def perspect(image,point,dst,shape):
    # 读入图片
    img = image
    src = point
    # 需要矫正成的形状，和上面一一对应

    # 获取矫正矩阵，也就步骤
    M = cv2.getPerspectiveTransform(src, dst)
    # 进行矫正，把img
    img = cv2.warpPerspective(img, M, shape)

    # 展示校正后的图形
    return np.array(img)

def crop_perspect(book_point, image):

    #width, height = (image.shape[1], image.shape[0])
    # shelf_point
    np_points = np.array(book_point, np.float32)
    # sort
    #np_points = BOOK_order_points_with_vitrual_center(np_points, width, height)
    w = np.int32(dist(np_points[0],np_points[1]))

    h = np.int32(dist(np_points[0], np_points[3]))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

    img_per = perspect(image, np_points, dst, (w, h))

    return img_per

#unique signal
def generate_unique_id():
    timestamp = str(int(time.time() * 1000)) # 时间戳
    uid = hashlib.md5(timestamp.encode()).hexdigest() # 使用MD5哈希函数生成唯一标识符
    return uid

def polygons_to_mask2(img_shape, polygons):
    '''
    边界点生成mask
    :param img_shape: [h,w]
    :param polygons: labelme JSON中的边界点格式 [[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]
    :return:
    '''
    mask = np.zeros(img_shape, dtype=np.uint8)
    polygons = np.asarray(polygons, np.int32) # 这里必须是int32，其他类型使用fillPoly会报错
    cv2.fillPoly(mask, [polygons], 1) # 非int32 会报错
    # cv2.fillConvexPoly(mask, polygons, 1)  # 非int32 会报错
    return mask

def crop_rec(book_dst, img_per_):

    # crop rectification
    # 切出
    mask = np.zeros_like(img_per_)
    # 将四点构成的区域填充为白色
    cv2.fillConvexPoly(mask, np.array(np.int32(book_dst)), (255, 255, 255))

    # 使用掩码裁剪
    cropped = cv2.bitwise_and(img_per_, mask)

    x1 = int(min(p[0] for p in book_dst))
    y1 = int(min(p[1] for p in book_dst))
    x2 = int(max(p[0] for p in book_dst))
    y2 = int(max(p[1] for p in book_dst))

    cropped1 = cropped[y1:y2, x1:x2]

    return cropped1

def filter_boxes(boxes: np.ndarray, keypoints, threshold=0.5):
    A = boxes.shape[0]
    keep = np.ones(A, dtype=bool)
    for i in range(A):
        if not keep[i]:
            continue
        for j in range(i+1, A):
            if not keep[j]:
                continue
            xy_max = np.minimum(boxes[i, 2:], boxes[j, 2:])
            xy_min = np.maximum(boxes[i, :2], boxes[j, :2])

            # 计算交集面积
            inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
            inter = inter[0]*inter[1]

            # 计算每个矩阵的面积
            area_i = (boxes[i, 2]-boxes[i, 0])*(boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[j, 2]-boxes[j, 0])*(boxes[j, 3] - boxes[j, 1])

            # 计算交并比
            iou = inter/(area_i+area_j-inter)

            # 如果交并比大于0.5，删除面积较小的边界框
            if iou > threshold:
                if area_i < area_j:
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    return keypoints[keep]

#获取最大外接矩形
def max_area_rect(keypoints):
    rects = []
    for quad in keypoints:
        quad = np.array(quad)
        x_min, y_min = np.min(quad, axis=0)
        x_max, y_max = np.max(quad, axis=0)
        rects.append([x_min, y_min, x_max, y_max])
    return np.array(rects)

#* width height
def scale_coordinates(keypoints, width, height):
    scaled_keypoints = []
    for quad in keypoints:
        scaled_quad = []
        for point in quad:
            x, y = point
            scaled_quad.append([x * width, y * height])
        scaled_keypoints.append(scaled_quad)
    return np.array(scaled_keypoints)

def sort_keypoints(keypoints):
    # 对每个四边形，取其所有点的x坐标的平均值作为排序依据
    sorted_keypoints = sorted(keypoints, key=lambda quad: np.mean([point[0] for point in quad]))
    return np.array(sorted_keypoints)

# 预测
def predict(image,parament):
    # 预处理图片
    img = preprocess(image)
    img = np.asarray(img)
    h, w, _ =img.shape
    #dir = make_file(img)
    print(parament)
    with torch.no_grad():
            result = model.predict(img, conf=float(parament['conf']), imgsz=640 ,iou=0.45, save_txt=False, save_crop=True, boxes=False, device='0')
            for r in result:
                keypoints = r.keypoints.xyn.cpu().numpy()
                list1 = []
                if np.size(keypoints) !=0:
                    keypoints = sort_keypoints(keypoints)
                    #scale expand
                    scaled_keypoints = scale_coordinates(keypoints, w, h)
                    for points in scaled_keypoints:
                        if len(points) != 0:
                            points = points[0:4]
                            cropped1 = crop_perspect(points, img)

                            # 从排序后的列表中提取图像，并将它们添加到新的列表中
                            list1.append(cropped1)

                else:
                        list1 = 'NONE'

            # 保存到本地
            # save_results(list1, dir)
            return list1

# 预处理
def preprocess(image):
    return Image.open(io.BytesIO(image))

# 返回base64
def transform(outputs):
    image_list = []
    if 'NONE' not in outputs:
        for img in outputs:
            # 转换为Image格式
            pil_img = Image.fromarray(img)
            # 编码为base64
            buff = io.BytesIO()
            pil_img.save(buff, format="PNG")
            img_str = base64.b64encode(buff.getvalue()).decode('utf-8')
            # 添加到列表
            image_list.append(img_str)

    else:
        image_list = None

    return image_list

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

host = '0.0.0.0'
port = 5003

app.run(host=host, port=port, debug=False)
