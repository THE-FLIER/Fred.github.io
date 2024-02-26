from PIL import Image
from ultralytics import YOLO
import os
import cv2
import numpy as np
import json
from sklearn.metrics.pairwise import euclidean_distances
import time
import math

input = 'dataset/rectangle_img'

#
output = 'results/1_26/lishui_crop'

# Load a pretrained YOLOv8n model
model = YOLO('/home/zf/yolov8/runs/pose/1_23_boos_13_points_addition2/weights/best.pt')
conf = 0.6

def _sorted(np_points,width, height):
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

    # shelf_point
    np_points = np.array(book_point, np.float32)
    # sort
    w = np.int32(dist(np_points[0], np_points[1]))

    h = np.int32(dist(np_points[0], np_points[3]))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

    img_per = perspect(image, np_points, dst, (w, h))

    return img_per


def polygons_to_mask2(img_shape, polygon):
    '''
    边界点生成mask
    :param img_shape: [h,w]
    :param polygon: labelme JSON中的边界点格式 [[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]
    :return:
    '''

    mask = np.zeros(img_shape, dtype=np.uint8)
    polygon = np.asarray(polygon, np.int32) # 这里必须是int32，其他类型使用fillPoly会报错
    # cv2.fillPoly(mask, polygons, 1) # 非int32 会报错
    cv2.fillConvexPoly(mask, polygon, 1)  # 非int32 会报错

    return mask

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
    return np.array([A, B, C, D])

def BOOK_order_points_with_vitrual_center(pts, width, height):
    pts = np.array(pts, dtype="float32")
    pts_ =pts
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])

    # 分为上下两组
    upper = pts[pts[:, 1] < center_y]
    lower = pts[pts[:, 1] >= center_y]

    # 在每组内部按照x值排序以分出左右
    upper_sorted = upper[np.argsort(upper[:, 0]), :]
    lower_sorted = lower[np.argsort(lower[:, 0]), :]

    # 确保上下两组都有两个点
    if upper_sorted.shape[0] != 2 or lower_sorted.shape[0] != 2:
        sorted_pts = _sorted(pts_, width, height)
        return sorted_pts

    # 合并左上、右上、右下、左下的点
    sorted_pts = np.array([upper_sorted[0], upper_sorted[1], lower_sorted[1], lower_sorted[0]], np.float32)
    return sorted_pts


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
    dir_list = [output, f"{output}/full", f"{output}/full_ori",f'{output}/visual/']
    for i in dir_list:
        os.makedirs(i, exist_ok=True)
    if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
        # Run inference on 'bus.jpg'
        source_path = os.path.join(input, file_name)
        ori_img = cv2.imread(source_path)

        h, w, _ = ori_img.shape

        start_time = time.time()
        results = model.predict(source_path, conf=conf, imgsz=640,iou=0.45, save_txt=False, save_crop=False, boxes=False, device='0')  # results list
        end_time = time.time()
        preprocess_time = round((end_time - start_time) * 1000,1)
        print(f"耗时： {preprocess_time}")
        a = file_name[:-4]
        filename = f'{a}.json'
        save = f'{output}/crop/{a}'
        os.makedirs(save, exist_ok=True)
        # Show the results
        for r in results:
            im_array = r.plot(conf=True)  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            # im.show()  # show image
            im.save(f'{output}/visual/{file_name}')
            keypoints = r.keypoints.xyn.cpu().numpy()
            if np.size(keypoints) != 0:
                keypoints = sort_keypoints(keypoints)
                c = 1
                black_img = np.zeros([h, w], dtype=np.uint8)
                res = np.zeros_like(ori_img)
                shapes = []

                scaled_keypoints = scale_coordinates(keypoints, w, h)


                ori_img_ = ori_img
                for points in scaled_keypoints:
                    if len(points) != 0:
                        points = points[0:4]
                        cropped1 = crop_perspect(points, ori_img)
                        cropped2 = crop_rec(points, ori_img)
                        # for i, point in enumerate(points):
                        #     cv2.circle(ori_img_, np.int32(point), 2, (255, 155, 255), 4)
                        #     cv2.putText(ori_img_, str(i + 1), np.int32(point), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        # # 保存结果
                        cv2.imwrite(f'{save}/{c}_p.jpg', cropped1)
                        cv2.imwrite(f'{save}/{c}_o.jpg', cropped2)

                        c += 1
                        mask_ = polygons_to_mask2([h, w], points)
                        res[mask_ > 0] = ori_img[mask_ > 0]

                        polygons = np.asarray(points, np.int32)
                        cv2.fillConvexPoly(black_img, polygons, 1)

                #cv2.imwrite(f"{output}/full/{a}.jpg", black_img * 255)
                cv2.imwrite(f"{output}/full_ori/{a}_ori.jpg", res)
                cv2.imwrite(f"{output}/{a}_ori.jpg", ori_img_)

            else:
                print('No Detection')