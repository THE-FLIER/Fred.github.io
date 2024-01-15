import numpy as np
import cv2
import math

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


def expand_polygon(vertices, scale_x=1.05, scale_y=1.1):
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

if __name__ == '__main__':

    coordinate = [[611.5, 807.4999999999999], [6749.000000000001, 1374.1666666666665], [6294.833333333334, 2761.6666666666665], [1090.6666666666665, 2190.833333333333]]
    # 图片
    img = cv2.imread('images/9997(1).jpg')
    # 矫正结果
    img_per = get_cop_M(coordinate, img)
    cv2.imwrite('images/9997(1)_1.jpg',img_per)
