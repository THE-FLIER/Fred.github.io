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

def expand_polygon(vertices, scale_x=1.05, scale_y=1.08):
    # 计算中心点
    center = [sum(vertex[i] for vertex in vertices) / len(vertices) for i in range(2)]

    # 创建一个新的顶点列表
    new_vertices = []

    # 对于每个顶点
    for vertex in vertices:
        # 计算向量
        vector = [vertex[i] - center[i] for i in range(2)]

        # 扩大向量
        vector = [vector[0] * scale_x, vector[1] * scale_y]

        # 计算新的顶点
        new_vertex = [center[i] + vector[i] for i in range(2)]

        new_vertices.append(new_vertex)

    return new_vertices

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def pic_sorted(np_points,width, height):
    width, height = width, height
    # left_bottom = [0, 0]
    # left_top = [0, height]
    # right_bottom = [width, 0]
    # right_top = [width, height]
    sorted_points = []
    np_points = np_points.tolist()
    dst = [[0, 0], [width, 0], [width, height],[0, height]]
    for p in dst:
        min_dist = float("inf")
        closest_point = None
        for q in np_points:
            d = dist(p, q)
            if d < min_dist:
                min_dist = d
                closest_point = q
        sorted_points.append(closest_point)


    return np.array(sorted_points, np.float32)

def order_points_with_vitrual_center(pts, width, height):
    pts = np.array(pts, dtype="float32")
    pts_ =pts
    center_x = np.mean(pts[:, 0])

    # 分为左右两组
    left = pts[pts[:, 0] < center_x]
    right = pts[pts[:, 0] >= center_x]

    # 在每组内部按照y值排序以分出上下
    left_sorted = left[np.argsort(left[:, 1]), :]
    right_sorted = right[np.argsort(right[:, 1]), :]

    # 确保左右两组都有两个点
    if left_sorted.shape[0] != 2 or right_sorted.shape[0] != 2:
        sorted_pts = pic_sorted(pts_, width, height)
        return sorted_pts
    # 合并左上、右上、右下、左下的点
    sorted_pts = np.array([left_sorted[0], right_sorted[0], right_sorted[1], left_sorted[1]], np.float32)
    return sorted_pts

def perspect(image,point):
    # 读入图片
    img = image
    src = point
    # 需要矫正成的形状，和上面一一对应
    dst = np.array([[0, 0], [3840, 0], [3840, 2160], [0, 2160]], np.float32)

    # 获取矫正矩阵，也就步骤
    M = cv2.getPerspectiveTransform(src, dst)
    # 进行矫正，把img
    img = cv2.warpPerspective(img, M, (3840, 2160))

    # 展示校正后的图形
    return np.array(img)

def get_cop_M(shelf_point,image):

    width, height = (image.shape[1], image.shape[0])
    # shelf_point
    np_points = np.array(shelf_point, np.float32)
    # sort
    np_points = order_points_with_vitrual_center(np_points, width, height)
    np_points_ = np.array(expand_polygon(np_points), np.float32)
    img_per= perspect(image, np_points_)

    return img_per

def calculate_centroid(points):
    return np.mean(points, axis=0)

def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def crop_image(points,img):
    x = points[:, 0]
    y = points[:, 1]
    y1 = int(min(y))
    y2 = int(max(y))
    x1 = int(min(x))
    x2 = int(max(x))

    mask = np.zeros_like(img)
    cv2.fillConvexPoly(mask, np.array(np.int32(points)), (255, 255, 255))
    crop = cv2.bitwise_and(img, mask)

    cropped = crop[y1:y2, x1:x2]

    return cropped

def sort_keypoints(keypoints):
    # 对每个四边形，取其所有点的x坐标的平均值作为排序依据
    sorted_keypoints = sorted(keypoints, key=lambda quad: np.mean([point[0] for point in quad]))
    return np.array(sorted_keypoints)

def scale_coordinates(keypoints, width, height):
    scaled_keypoints = []
    for quad in keypoints:
        scaled_quad = []
        for point in quad:
            x, y = point
            scaled_quad.append([x * width, y * height])
        scaled_keypoints.append(scaled_quad)
    return np.array(scaled_keypoints)

# 预测
def predict(image,parament):
    # 预处理图片
    img = preprocess(image)
    img = np.asarray(img)
    h, w, _ =img.shape
    dir = make_file(img)
    coordinate = parament['coordinate']
    list1 = []
    print(parament)
    if coordinate == None:
        with torch.no_grad():
                #INFERENCE
                result = model.predict(img, conf=float(parament['conf']), imgsz=640, save_txt=False, save_crop=True, boxes=False, device='0')
                for r in result:
                    keypoints = r.keypoints.xyn.cpu().numpy()
                    if np.size(keypoints) != 0:
                    # scale expand
                        scaled_keypoints = scale_coordinates(keypoints, w, h)
                        for points in scaled_keypoints:
                            points = points[0:4]
                            list1.append(points.tolist())
                    else:
                        list1 = 'None'

    else:
        img_per = get_cop_M(coordinate, img)
        list1 = img_per

    # save_results(list1, dir)
    return list1

#判断数组/列表
def is_np(s):
    try:
        s.shape
        return True
    except AttributeError:
        return False

# 预处理
def preprocess(image):
    return Image.open(io.BytesIO(image))

# 返回base64
def transform(outputs):
    if 'None' not in outputs:
        image_list = outputs.tolist()
    else:
        image_list = None

    return image_list

def transform_point(outputs):
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
    #result = transform(result)

    return jsonify({'content': result})

host = os.environ.get('APP_HOST')
port = os.environ.get('APP_PORT')

app.run(host=host, port=port, debug=False)
