#lambelme  标 转  yolov
import json
import os
import cv2
import numpy as np
import math
'''
会在同一目录下生成txt训练文件
'''
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

#透视变换转换
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
    return np.array(img), M

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

#找顺序
def order_points_with_vitrual_center(pts, width, height):
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

#排序
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

#扩大
def expand_polygon(vertices, scale_x=1.03, scale_y=1.08):
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

#判断书籍是否在书格内部
def is_inside(rect1, rect2):
    # 计算每个矩形的“左上角”和“右下角”
    left_top1 = (min(x for x, y in rect1), min(y for x, y in rect1))
    right_bottom1 = (max(x for x, y in rect1), max(y for x, y in rect1))
    left_top2 = (min(x for x, y in rect2), min(y for x, y in rect2))
    right_bottom2 = (max(x for x, y in rect2), max(y for x, y in rect2))

    # 比较“左上角”和“右下角”
    if (left_top1[0] >= left_top2[0] and left_top1[1] >= left_top2[1] and
            right_bottom1[0] <= right_bottom2[0] and right_bottom1[1] <= right_bottom2[1]):
        return True
    else:
        return False
def get_cop_M(shelf_point,image):

    width, height = (image.shape[1], image.shape[0])
    # shelf_point
    np_points = np.array(shelf_point, np.float32)
    # sort
    np_points = order_points_with_vitrual_center(np_points, width, height)
    np_points_ = np.array(expand_polygon(np_points), np.float32)
    img_per, M = perspect(image, np_points_)

    return img_per, M

#json_read
def lambelme_json_label_to_yolov_seg_label(json_path,crop_path):
    import glob
    import numpy as np
    json_path = json_path
    json_files = glob.glob(json_path + "/*.json")
    for json_file in json_files:
        # if json_file != r"C:\Users\jianming_ge\Desktop\code\handle_dataset\water_street\223.json":
        #     continue
        print(json_file)
        f = open(json_file, 'rb')
        json_info = json.load(f)
        a = 1
        # print(json_info.keys())
        #img = cv2.imread(os.path.join(json_path, json_info["imagePath"]))
        height = json_info['imageHeight']
        width = json_info['imageWidth']
        np_w_h = np.array([[width, height]], np.int32)
        txt_file = os.path.basename(json_file)
        image_file = f'datasets/book-point/test/{txt_file[:-5]}.JPG'

        shelves_list = []
        book_list = []
        for point_json in json_info["shapes"]:
            if is_number(point_json["label"]):
                if float(point_json["label"]) in [0.0, 1.0]:
                    np_points = np.array(point_json["points"], np.float32)
                    shelves_list.append(np_points)
                elif float(point_json["label"]) == 4.0 :
                    np_points = np.array(point_json["points"], np.float32)
                    book_list.append(np_points)

        image = cv2.imread(image_file)
        for shelves in shelves_list:
            save_path = os.path.join(crop_path, f'{txt_file[:-5]}_{a}')
            img_per, matrix = get_cop_M(shelves, image)
            img_per_ = img_per.copy()

            width, height = (img_per.shape[1], img_per.shape[0])
            np_w_h = np.array([[width, height]], np.int32)
            # image save
            cv2.imwrite(f'{save_path}.jpg', img_per)
            # file save
            txt_file_ = f'{save_path}.txt'
            f = open(txt_file_, "w")

            # 对内部每个书本框进行透视转换
            for book in book_list:
                txt_con = ""
                if is_inside(book, shelves):
                    #坐标变换
                    book = book.reshape(-1, 1, 2)
                    book_dst = cv2.perspectiveTransform(book, matrix)
                    book_dst = book_dst.reshape(-1, 2)
                    # keypoint
                    min_x = min(book_dst, key=lambda point: point[0])[0]
                    max_x = max(book_dst, key=lambda point: point[0])[0]
                    min_y = min(book_dst, key=lambda point: point[1])[1]
                    max_y = max(book_dst, key=lambda point: point[1])[1]

                    # 计算外接矩形的宽度和高度
                    width_ = (max_x - min_x) / width
                    height_ = (max_y - min_y) / height

                    # 计算外接矩形的中心点
                    center_x = ((min_x + max_x) / 2) / width
                    center_y = ((min_y + max_y) / 2) / height
                    txt_con += f'0 {center_x} {center_y} {width_} {height_} '
                    norm_points = book_dst / np_w_h
                    norm_points_list = norm_points.tolist()
                    txt_content = f"{txt_con}" + " ".join(
                        [" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"

                    cv2.polylines(img_per_, [np.int32(book_dst)], True, (0, 255, 0), 2)

                    f.write(txt_content)

            #cv2.imwrite(f'trans/{txt_file[:-5]}_{a}.jpg', img_per_)
            a += 1

                    # 保存图像
                    #cv2.polylines(image_roi, [np_points_], True, (0, 255, 0), 2)
                    #box
                    # image_roi_ = image_roi.copy()
                    # cv2.polylines(image_roi, [rectangle_points], True, (0, 255, 0), 2)
                    # cv2.imwrite(f'{crop_path}/ori/{txt_file[:-5]}_{a}.jpg', image_roi)

                    #变换
                    # img_per = perspect(image_roi, np_points)
                    # cv2.imwrite(f'{crop_path}/crop/{txt_file[:-5]}_{a}.jpg', img_per)
                    # a +=1


if __name__=="__main__":
    json_path = "datasets/book-point/test"
    crop_path = "datasets/mm_book/test"
    lambelme_json_label_to_yolov_seg_label(json_path, crop_path)