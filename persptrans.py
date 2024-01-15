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
def expand_polygon(vertices, width, height, scale_x=1.08, scale_y=1.1):
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

        # 检查新的顶点是否超出了图片的边界
        new_vertex = [min(max(new_vertex[i], 0), [width, height][i]) for i in range(2)]

        new_vertices.append(new_vertex)

    return new_vertices

#判断书籍是否在书格内部
def is_inside(polygon1, polygon2):
    def is_point_inside_polygon(point, polygon):
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    return all(is_point_inside_polygon(vertex, polygon2) for vertex in polygon1)

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

#get martin
def get_cop_M(shelf_point,image):

    width, height = (image.shape[1], image.shape[0])
    # shelf_point
    np_points = np.array(shelf_point, np.float32)
    # sort
    np_points = order_points_with_vitrual_center(np_points, width, height)
    np_points_ = np.array(expand_polygon(np_points, width, height), np.float32)
    img_per, M = perspect(image, np_points_)

    return img_per, M, np_points_

def txt_make(width, height, np_points):

    width, height = (width, height)
    ori_np_w_h = np.array([[width, height]], np.int32)

    min_x = min(np_points, key=lambda point: point[0])[0]
    max_x = max(np_points, key=lambda point: point[0])[0]
    min_y = min(np_points, key=lambda point: point[1])[1]
    max_y = max(np_points, key=lambda point: point[1])[1]

    # 计算外接矩形的宽度和高度
    width_ = (max_x - min_x) / width
    height_ = (max_y - min_y) / height
    # 计算外接矩形的中心点
    center_x = ((min_x + max_x) / 2) / width
    center_y = ((min_y + max_y) / 2) / height

    txt_con = ""
    # shelves txt write
    txt_con += f'0 {center_x} {center_y} {width_} {height_} '
    norm_points = np_points / ori_np_w_h
    norm_points_list = norm_points.tolist()
    txt_content = f"{txt_con}" + " ".join(
        [" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"

    return txt_content

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

#json_read
def lambelme_json_label_to_yolov_seg_label(image_path, json4book_path, json4shelve_path,crop_path):
    import glob
    import numpy as np
    book_path = json4book_path
    shelve_path = glob.glob(json4shelve_path + "/*.json")
    for jsonfile in shelve_path:
        a = 1
        b = 1
        pure_name = os.path.basename(jsonfile)
        pure_name = pure_name[:-5]
        #json
        book_spine_path = os.path.join(book_path, f'{pure_name}.json')
        book_shelf_path = jsonfile
        images = os.path.join(image_path, f'{pure_name}.jpg')
        print(f'{pure_name} start')
        f1_book = open(book_spine_path, 'rb')
        book_info = json.load(f1_book)

        f2_shelf = open(book_shelf_path, 'rb')
        shelf_info = json.load(f2_shelf)

        ori_height = book_info['imageHeight']
        ori_width = book_info['imageWidth']

        shelves_list = []
        book_list = []

        image = cv2.imread(images)

        #book_spine extract
        for point_json in book_info["shapes"]:
            np_points = np.array(point_json["points"], np.float32)
            if len(np_points) ==4:
                book_list.append(np_points)

        #book_shelf extract
        for point_json in shelf_info["shapes"]:
            np_points = np.array(point_json["points"], np.float32)
            if len(np_points) == 4:
                shelves_list.append(np_points)

        ori_book_image_save_path_ = os.path.join(crop_path, 'crop_ori')
        os.makedirs(ori_book_image_save_path_, exist_ok=True)

        #ori_save
        for ori_books in book_list:
            points = np.int32(ori_books)
            crop = crop_rec(points, image)
            # 保存
            cv2.imwrite(f"{ori_book_image_save_path_}/{pure_name}_{b}.jpg", crop)
            b += 1

        d = 1
        os.makedirs('./trans', exist_ok=True)

        #遍历 shelves
        for shelves in shelves_list:
            #save path create
            #train txt for shelves
            # image_ = image.copy()
            # cv2.polylines(image_, [np.int32(shelves)], True, (0, 0, 255), 2)

            shelf_save_path = os.path.join(crop_path, 'book_shelf/', pure_name)
            shelf_save_path_ = os.path.join(crop_path, 'book_shelf/')

            #train txt for book
            book_txt_save_path = os.path.join(crop_path, 'book_spine/', pure_name)
            book_txt_save_path_ = os.path.join(crop_path, 'book_spine/')
            #train  for image
            book_image_save_path = book_txt_save_path

            #json for mm
            book_json_save_path_ = os.path.join(crop_path, 'coco_book/')
            book_json_save_path = os.path.join(crop_path, 'coco_book/', pure_name)
            path_create = [shelf_save_path_, book_txt_save_path_,book_json_save_path_]
            for path in path_create:
                os.makedirs(path, exist_ok=True)

            # shelf txt create
            shelf_file = f'{shelf_save_path}.txt'
            f_shelf_txt = open(shelf_file, "a")
            txt_content = txt_make(ori_width, ori_height, shelves)
            f_shelf_txt.write(txt_content)

            #rectificate image
            img_per, matrix, fix_shelf = get_cop_M(shelves, image)
            img_per_ = img_per.copy()
            # crop_image save
            cv2.imwrite(f'{book_image_save_path}_{a}.jpg', img_per)

            # book txt create
            book_file = f'{book_txt_save_path}_{a}.txt'
            f_book_txt = open(book_file, "w")
            c = 1
            # 对内部每个书本框进行透视转换

            rectify_book_image_save_path = os.path.join(crop_path, f'crop_rectify/{pure_name}')
            rectify_book_image_save_path_ = os.path.join(crop_path, f'crop_rectify')
            os.makedirs(rectify_book_image_save_path_, exist_ok=True)

            book_shapes = []
            cv2.imwrite(f'{book_json_save_path}_{a}.jpg', img_per)
            width, height = (img_per.shape[1], img_per.shape[0])

            for book in book_list:
                    if is_inside(book, fix_shelf):
                        # 坐标变换
                        # and len(book) == 4
                        book = book.reshape(-1, 1, 2)
                        book_dst = cv2.perspectiveTransform(book, matrix)
                        book_dst = book_dst.reshape(-1, 2)
                        #cv2.polylines(img_per_, [np.int32(book_dst)], True, (255, 0, 0), 2)
                        txt_con = txt_make(width, height, book_dst)
                        f_book_txt.write(txt_con)
                        book_shapes.append({"label": "1", "points": book_dst.tolist()})
                        # image save
                        cropped1 = crop_rec(book_dst, img_per_)
                        if np.size(cropped1) != 0:
                            cv2.imwrite(f"{rectify_book_image_save_path}_{c}.jpg", cropped1)
                            c += 1

            book_data = {
                "version": "5.2.1",
                "flags": {},
                "shapes": book_shapes,
                "imagePath": f'{pure_name}_{a}.jpg',
                "imageData": None,
                "imageWidth": width,
                "imageHeight": height
            }

            write_to_json(book_json_save_path_, f'{pure_name}_{a}', book_data)

            #cv2.imwrite(f'trans/{txt_file[:-5]}_{a}.jpg', img_per_)
            a += 1
            cv2.imwrite(f'./trans/{pure_name}_rectify_{d}.jpg', img_per_)
            d += 1

        print(f'{pure_name} finished')

if __name__ == "__main__":
    image_path = "dataset/ori/1000/img"
    json4book_path = "dataset/ori/1000/book"
    json4shelve_path = "dataset/ori/1000/shelf"
    crop_path = "dataset/ori/1000/"
    lambelme_json_label_to_yolov_seg_label(image_path, json4book_path, json4shelve_path, crop_path)