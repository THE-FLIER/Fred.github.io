#lambelme  标 转  yolov
import json
import os
import cv2
import numpy as np
from test import dataset_split
import shutil
import math
'''
会在同一目录下生成txt训练文件
'''

def calculate_points_21(np_points):
    A, B, C, D = np_points[0],np_points[1],np_points[2],np_points[3]
    # 计算相邻两点的中点坐标
    E = ((A[0]+B[0])/2, (A[1]+B[1])/2)
    F = ((B[0]+C[0])/2, (B[1]+C[1])/2)
    G = ((C[0]+D[0])/2, (C[1]+D[1])/2)
    H = ((D[0]+A[0])/2, (D[1]+A[1])/2)

    # 计算对角线交点坐标
    I = ((A[0]+C[0])/2, (A[1]+C[1])/2)

    # 计算前八个点每一个和对角线的中点坐标
    J = ((A[0]+I[0])/2, (A[1]+I[1])/2)
    K = ((B[0]+I[0])/2, (B[1]+I[1])/2)
    L = ((C[0]+I[0])/2, (C[1]+I[1])/2)
    M = ((D[0]+I[0])/2, (D[1]+I[1])/2)
    N = ((E[0]+I[0])/2, (E[1]+I[1])/2)
    O = ((F[0]+I[0])/2, (F[1]+I[1])/2)
    P = ((G[0]+I[0])/2, (G[1]+I[1])/2)
    Q = ((H[0]+I[0])/2, (H[1]+I[1])/2)

    R = ((E[0]+N[0])/2, (E[1]+N[1])/2)
    S = ((N[0]+I[0])/2, (N[1]+I[1])/2)
    T = ((P[0]+I[0])/2, (P[1]+I[1])/2)
    U = ((G[0]+P[0])/2, (G[1]+P[1])/2)
    return np.array([A, B, C, D, E, F, G, H,J, K, L, M, N, O, P, Q, R, S, I, T, U])

def calculate_points_11(np_points):
    A, B, C, D = np_points[0],np_points[1],np_points[2],np_points[3]
    # 计算相邻两点的中点坐标
    E = ((A[0]+B[0])/2, (A[1]+B[1])/2)
    F = ((C[0]+D[0])/2, (C[1]+D[1])/2)
    # 计算对角线交点坐标
    I = ((A[0]+C[0])/2, (A[1]+C[1])/2)

    # 计算前八个点每一个和对角线的中点坐标
    G = ((E[0] + I[0]) /2, (E[1] + I[1]) /2)
    H = ((E[0] + I[0]) /2, (E[1] + I[1]) /2)
    J = ((F[0] + I[0])/2, (F[1] + I[1]) /2)
    K = ((F[0] + I[0]) /2, (F[1] + I[1])/2)

    return np.array([A, B, C, D, E, F, G, H,J, K, I])

def calculate_points_13(np_points):
    A, B, C, D = np_points[0],np_points[1],np_points[2],np_points[3]
    # 计算相邻两点的中点坐标
    E = ((A[0]+B[0])/2, (A[1]+B[1])/2)
    F = ((B[0]+C[0])/2, (B[1]+C[1])/2)
    G = ((C[0]+D[0])/2, (C[1]+D[1])/2)
    H = ((D[0]+A[0])/2, (D[1]+A[1])/2)
    # 计算对角线交点坐标
    I = ((A[0]+C[0])/2, (A[1]+C[1])/2)

    # 计算前八个点每一个和对角线的中点坐标
    J = ((A[0]+H[0])/2, (A[1]+H[1])/2)
    K = ((B[0]+F[0])/2, (B[1]+F[1])/2)
    L = ((C[0]+F[0])/2, (C[1]+F[1])/2)
    M = ((D[0]+H[0])/2, (D[1]+H[1])/2)

    return np.array([A, B, C, D, E, F, G, H,J, K, L, M, I])

def calculate_shelves_points_13(np_points):
    A, B, C, D = np_points[0],np_points[1],np_points[2],np_points[3]
    # 计算相邻两点的中点坐标
    E = ((A[0]+B[0])/2, (A[1]+B[1])/2)
    F = ((B[0]+C[0])/2, (B[1]+C[1])/2)
    G = ((C[0]+D[0])/2, (C[1]+D[1])/2)
    H = ((D[0]+A[0])/2, (D[1]+A[1])/2)
    # 计算对角线交点坐标
    I = ((A[0]+C[0])/2, (A[1]+C[1])/2)

    # 计算前八个点每一个和对角线的中点坐标
    J = ((A[0]+E[0])/2, (A[1]+E[1])/2)
    K = ((B[0]+E[0])/2, (B[1]+E[1])/2)
    L = ((C[0]+G[0])/2, (C[1]+G[1])/2)
    M = ((D[0]+G[0])/2, (D[1]+G[1])/2)

    return np.array([A, B, C, D, E, F, G, H,J, K, L, M, I])

def obb(np_points):
    A, B, C, D = np_points[0], np_points[1], np_points[2], np_points[3]

    return np.array([A, B, C, D])


def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def sorted(np_points,width, height):
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
        sorted_pts = sorted(pts_, width, height)
        return sorted_pts
    # 合并左上、右上、右下、左下的点
    sorted_pts = np.array([upper_sorted[0], upper_sorted[1], lower_sorted[1], lower_sorted[0]], np.float32)
    return sorted_pts

def SHELVES_order_points_with_vitrual_center(pts, width, height):
    pts = np.array(pts, dtype="float32")
    pts_ =pts
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])

    # 分为左右两组
    left = pts[pts[:, 0] < center_x]
    right = pts[pts[:, 0] >= center_x]

    # 在每组内部按照y值排序以分出上下
    left_sorted = left[np.argsort(left[:, 1]), :]
    right_sorted = right[np.argsort(right[:, 1]), :]

    # 确保左右两组都有两个点
    if left_sorted.shape[0] != 2 or right_sorted.shape[0] != 2:
        sorted_pts = sorted(pts_, width, height)
        return sorted_pts
    # 合并左上、右上、右下、左下的点
    sorted_pts = np.array([left_sorted[0], right_sorted[0], right_sorted[1], left_sorted[1]], np.float32)
    return sorted_pts

def make_points(np_points,np_w_h):
    width, height = np_w_h[0][0],np_w_h[0][1]
    #BOOK
    np_points = BOOK_order_points_with_vitrual_center(np_points, width, height)

    # SHELVES
    #np_points = SHELVES_order_points_with_vitrual_center(np_points, width, height)

    elven_points = calculate_points_13(np_points)
    elven_points = elven_points / np_w_h
    elven_points = elven_points.tolist()

    return elven_points

def lambelme_json_label_to_yolov_seg_label(json_path,txt_save):
    import glob
    import numpy as np
    json_path = json_path
    json_files = glob.glob(json_path + "/*.json")
    for json_file in json_files:
        # if json_file != r"C:\Users\jianming_ge\Desktop\code\handle_dataset\water_street\223.json":
        #     continue
        print(json_file)
        f = open(json_file,'rb')
        json_info = json.load(f)
        # print(json_info.keys())
        #img = cv2.imread(os.path.join(json_path, json_info["imagePath"]))
        height = json_info['imageHeight']
        width = json_info['imageWidth']
        np_w_h = np.array([[width, height]], np.int32)
        txt_file = os.path.basename(json_file)
        txt_file_ = f'{txt_save}/{txt_file[:-5]}.txt'
        img_path = f'{json_path}/{txt_file[:-5]}.jpg'
        img = cv2.imread(img_path)
        f = open(txt_file_, "w")
        for point_json in json_info["shapes"]:

            #seg
            # txt_content = ""
            # np_points = np.array(point_json["points"], np.int32)
            # norm_points = np_points / np_w_h
            # norm_points_list = norm_points.tolist()
            # txt_content += "0 " + " ".join([" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
            # f.write(txt_content)

            #keypoints
            np_points = np.array(point_json["points"], np.int32)

            if len(np_points) == 4:

                min_x = min(np_points, key=lambda point: point[0])[0] + 0.5
                max_x = max(np_points, key=lambda point: point[0])[0] + 0.5
                min_y = min(np_points, key=lambda point: point[1])[1] + 0.5
                max_y = max(np_points, key=lambda point: point[1])[1] + 0.5
                # 计算外接矩形的宽度和高度
                width_ = float(max_x - min_x) / float(width)
                height_ = float(max_y - min_y) / float(height)
                # 计算外接矩形的中心点
                center_x = ((min_x + max_x) / 2) / float(width)
                center_y = ((min_y + max_y) / 2) / float(height)


                norm_points_list= make_points(np_points,np_w_h)

                txt_con = ""
                txt_con += f'0 {center_x} {center_y} {width_} {height_} '
                txt_content = f"{txt_con}" + " ".join(
                    [" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"

                #norm_points_list = [[center_x, center_y], [width_, height_]]
                # txt_content = f"0 " + " ".join(
                #     [" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
                #
                f.write(txt_content)

                #obb
                # norm_points_list = make_points(np_points, np_w_h)
                # txt_content = f"0 " + " ".join(
                #     [" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
                # f.write(txt_content)

        #         for i, point in enumerate(norm_points_list):
        #             cv2.circle(img, np.int32(point), 2, (255, 155, 255), 4)
        #             cv2.putText(img, str(i), np.int32(point), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #
        # cv2.imwrite(f'dataset/1000_shelves/{txt_file[:-5]}.jpg',img)

if __name__=="__main__":
    #labelme json文件位置
    json_path = "dataset/book_1_23"

    #数据集结构
    txt_save = 'dataset/multi_points/extracted_13/labels/train'
    src_folder = 'dataset/multi_points/extracted_13/images/train'
    dst_folder = 'dataset/multi_points/extracted_13/images/val'

    # 定义源标签文件夹和目标标签文件夹进行数据集划分8：2
    src_label_folder = 'dataset/multi_points/extracted_13/labels/train'
    dst_label_folder = 'dataset/multi_points/extracted_13/labels/val'
    path_ = [src_folder,dst_folder,src_label_folder,dst_label_folder]
    for i in path_:
        os.makedirs(i,exist_ok=True)

    for filename in os.listdir(json_path):
        if filename.endswith(".jpg"):
            shutil.copy(os.path.join(json_path, filename), src_folder)

    lambelme_json_label_to_yolov_seg_label(json_path,txt_save)

    # 定义源文件夹和目标文件夹
    dataset_split(src_folder,dst_folder,src_label_folder,dst_label_folder)