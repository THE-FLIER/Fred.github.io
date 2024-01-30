from PIL import Image
from ultralytics import YOLO
import os
import cv2
import numpy as np
import json
from sklearn.metrics.pairwise import euclidean_distances

input = 'dataset/book_detect_val'

#
output = 'results/1_24/book_detect_0.3'

# Load a pretrained YOLOv8n model
model = YOLO('/home/zf/yolov8/runs/detect/1-18-book-det/weights/best.pt')
conf = 0.3

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

def calculate_fourth_point(points):
    A = points[0]
    B = points[1]
    C = points[2]

    x4 = A[0] + (C[0] - B[0])
    y4 = C[1] + (A[1] - B[1])
    D = [x4, y4]
    return np.array([A,B,C,D])

def write_to_json(directory, filename, data):
    """
    Check if a JSON file with the specified filename exists in the given directory.
    If not, create it and write the provided data to it.

    :param directory: The directory to check or create the file in.
    :param filename: The name of the file to check or create.
    :param data: A dictionary containing the data to write to the JSON file.
    """
    # Ensure the filename ends with .json
    if not filename.endswith('.json'):
        filename += '.json'

    # Construct the full path
    file_path = os.path.join(directory, filename)

    # Check if file exists
    if not os.path.exists(file_path):
        # Create the file and write the data
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"File '{filename}' created in '{directory}'.")
    else:
        print(f"File '{filename}' already exists in '{directory}'.")

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

for file_name in os.listdir(input):
    dir_list = [output,  f"{output}/full_ori",f'{output}/visual/']
    for i in dir_list:
        os.makedirs(i, exist_ok=True)
    if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
        # Run inference on 'bus.jpg'
        source_path = os.path.join(input, file_name)
        ori_img = cv2.imread(source_path)
        h, w, _ = ori_img.shape
        results = model.predict(source_path, conf=conf, imgsz=640, save_txt=False, save_crop=False, boxes=False, device='0')  # results list
        # Show the results
        for r in results:
            im_array = r.plot(conf=True)  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            # im.show()  # show image
            im.save(f'{output}/visual/{file_name}')
