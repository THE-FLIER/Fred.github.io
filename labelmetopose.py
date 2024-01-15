import os
import sys
import glob
import json
import shutil
import argparse
import numpy as np
import PIL.Image
import os.path as osp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pycocotools import mask
import math
class Labelme2coco_keypoints():
    def __init__(self, args):
        """
        Lableme 关键点数据集转 COCO 数据集的构造函数:

        Args
            args：命令行输入的参数
                - class_name 根类名字

        """

        self.classname_to_id = {args.class_name: 1}
        self.images = []
        self.annotations = []
        self.categories = []
        self.ann_id = 0
        self.img_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    def _get_box(self, points):

        np_points = np.array(points)

        xmin = np.min(np_points[:, 0])
        xmax = np.max(np_points[:, 0])
        ymin = np.min(np_points[:, 1])
        ymax = np.max(np_points[:, 1])


        # 计算宽度和高度
        w = xmax - xmin
        h = ymax - ymin
        return [xmin,ymin,w,h]

    def _get_keypoints(self, points, keypoints, num_keypoints):
        """
        解析 labelme 的原始数据， 生成 coco 标注的 关键点对象

        例如：
            "keypoints": [
                67.06149888292556,  # x 的值
                122.5043507571318,  # y 的值
                1,                  # 相当于 Z 值，如果是2D关键点 0：不可见 1：表示可见。
                82.42582269256718,
                109.95672933232304,
                1,
                ...,
            ],

        """

        if points[0] == 0 and points[1] == 0:
            visable = 0
        else:
            visable = 2
            num_keypoints += 1

        keypoints.extend([points[0], points[1], visable])
        return keypoints, num_keypoints

    def _calculate_points_13(self,np_points):

        A, B, C, D = np_points[0], np_points[1], np_points[2], np_points[3]
        # 计算相邻两点的中点坐标
        E = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)
        F = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2)
        G = ((C[0] + D[0]) / 2, (C[1] + D[1]) / 2)
        H = ((D[0] + A[0]) / 2, (D[1] + A[1]) / 2)
        # 计算对角线交点坐标
        I = ((A[0] + C[0]) / 2, (A[1] + C[1]) / 2)

        # 计算前八个点每一个和对角线的中点坐标
        J = ((A[0] + H[0]) / 2, (A[1] + H[1]) / 2)
        K = ((B[0] + F[0]) / 2, (B[1] + F[1]) / 2)
        L = ((C[0] + F[0]) / 2, (C[1] + F[1]) / 2)
        M = ((D[0] + H[0]) / 2, (D[1] + H[1]) / 2)

        return np.array([A,B,C,D,E,F,G,H,J,K,L,M,I])


    def __cal_area(self,points):
        np_points = np.array(points)

        xmin = np.min(np_points[:, 0])
        xmax = np.max(np_points[:, 0])
        ymin = np.min(np_points[:, 1])
        ymax = np.max(np_points[:, 1])

        # 计算宽度和高度
        w = xmax - xmin
        h = ymax - ymin
        return w*h
    def _cal_area(self,seg,h,w):

        rle = mask.frPyObjects(seg,
                               h,
                               w)
        # 计算面积
        area = mask.area(rle)

        return int(area[0])

    # 从注释中获取RLE掩码
    def dist(self,p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def sorted(self,np_points, width, height):
        width, height = width, height
        # left_bottom = [0, 0]
        # left_top = [0, height]
        # right_bottom = [width, 0]
        # right_top = [width, height]
        sorted_points = []
        np_points = np_points.tolist()
        dst = [[0, 0], [width, 0], [width, height], [0, height]]
        for p in dst:
            min_dist = float("inf")
            closest_point = None
            for q in np_points:
                d = self.dist(p, q)
                if d < min_dist:
                    min_dist = d
                    closest_point = q
            sorted_points.append(closest_point)

        return np.array(sorted_points, np.float32)

    def order_points_with_vitrual_center(self,pts, width, height):
        pts = np.array(pts, dtype="float32")
        pts_ = pts
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
            sorted_pts = self.sorted(pts_, width, height)
            return sorted_pts
        # 合并左上、右上、右下、左下的点
        sorted_pts = np.array([upper_sorted[0], upper_sorted[1], lower_sorted[1], lower_sorted[0]], np.float32)
        return sorted_pts

    def _image(self, obj, path):
        """
        解析 labelme 的 obj 对象，生成 coco 的 image 对象

        生成包括：id，file_name，height，width 4个属性

        示例：
             {
                "file_name": "training/rgb/00031426.jpg",
                "height": 224,
                "width": 224,
                "id": 31426
            }

        """

        image = {}

        # img_x = utils.img_b64_to_arr(obj['imageData'])  # 获得原始 labelme 标签的 imageData 属性，并通过 labelme 的工具方法转成 array
        # image['height'], image['width'] = img_x.shape[:-1]  # 获得图片的宽高

        #
        image['height'], image['width'] = obj['imageHeight'], obj['imageWidth']
        # self.img_id = int(os.path.basename(path).split(".json")[0])

        image['id'] = self.img_id

        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")

        return image

    def _annotation(self, bboxes_list, keypoints_list, json_path,h,w):
        """
        生成coco标注

        Args：
            bboxes_list： 矩形标注框
            keypoints_list： 关键点
            json_path：json文件路径

        """

        # if len(keypoints_list) != args.join_num * len(bboxes_list):
        #     print('you loss {} keypoint(s) with file {}'.format(args.join_num * len(bboxes_list) - len(keypoints_list), json_path))
        #     print('Please check ！！！')
        #     sys.exit()
        i = 0
        for object in bboxes_list:
            annotation = {}
            keypoints = []
            num_keypoints = 0
            label = object['label']
            bbox = object['points']
            annotation['id'] = self.ann_id
            annotation['image_id'] = self.img_id
            annotation['category_id'] = 1
            annotation['iscrowd'] = 0
            annotation['segmentation'] = [np.asarray(bbox).flatten().tolist()]
            annotation['area'] = self._cal_area(annotation['segmentation'],h,w)
            annotation['bbox'] = self._get_box(bbox)

            for keypoint in keypoints_list[i * args.join_num: (i + 1) * args.join_num]:
                point = keypoint['points']
                np_points = self.order_points_with_vitrual_center(point, w, h)
                np_points = np_points.tolist()
                # np_points = self._calculate_points_13(np_points)
                if len(np_points) ==4:
                    for p in np_points:
                        annotation['keypoints'], num_keypoints = self._get_keypoints(p, keypoints, num_keypoints)
                # extend_point = [0.0,0.0,0]
                # for _ in range(13):
                #     kpoints.extend(extend_point)
                # annotation['keypoints'] = kpoints

            annotation['num_keypoints'] = num_keypoints


            self.ann_id += 1
            self.annotations.append(annotation)
            i += 1

    def _init_categories(self):
        """
        初始化 COCO 的 标注类别

        例如：
        "categories": [
            {
                "supercategory": "hand",
                "id": 1,
                "name": "hand",
                "keypoints": [
                    "wrist",
                    "thumb1",
                    "thumb2",
                    ...,
                ],
                "skeleton": [
                ]
            }
        ]
        """

        for name, id in self.classname_to_id.items():
            kepoints =['left_top','right_top','right_bottom','left_bottom']
            skeleton=[[0,1],[1,2],[2,3],[3,0]]
            category = {}
            category['supercategory'] = name
            category['id'] = id
            category['name'] = name
            # 21 个关键点数据
            # category['keypoint'] = kepoints
            # category['skeleton'] = skeleton
            # category['keypoint'] = [str(i + 1) for i in range(args.join_num)]


            self.categories.append(category)

    def to_coco(self, json_path_list):
        """
        Labelme 原始标签转换成 coco 数据集格式，生成的包括标签和图像

        Args：
            json_path_list：原始数据集的目录

        """

        self._init_categories()
        i = 0
        for json_path in tqdm(json_path_list):
            obj = self.read_jsonfile(json_path)  # 解析一个标注文件
            self.images.append(self._image(obj, json_path))  # 解析图片
            shapes = obj['shapes']  # 读取 labelme shape 标注

            bboxes_list, keypoints_list = [], []
            for shape in shapes:
                    bboxes_list.append(shape)
                    keypoints_list.append(shape)# keypoints
                # elif shape['shape_type'] == 'polygon':

            h,w = self.images[i]['height'], self.images[i]['width']
            self._annotation(bboxes_list, keypoints_list, json_path,h,w)
            self.img_id = self.img_id + 1
            i +=1

        keypoints = {}
        keypoints['info'] = {'description': 'LablemeDataset', 'version': 1.0, 'year': 2023}
        keypoints['license'] = ['BUAA']
        keypoints['images'] = self.images
        keypoints['annotations'] = self.annotations
        keypoints['categories'] = self.categories
        return keypoints

def init_dir(base_path):
    """
    初始化COCO数据集的文件夹结构；
    coco - annotations  #标注文件路径
         - train        #训练数据集
         - val          #验证数据集
    Args：
        base_path：数据集放置的根路径
    """
    if not os.path.exists(os.path.join(base_path, "coco", "annotations")):
        os.makedirs(os.path.join(base_path, "coco", "annotations"))
    if not os.path.exists(os.path.join(base_path, "coco", "train")):
        os.makedirs(os.path.join(base_path, "coco", "train"))
    if not os.path.exists(os.path.join(base_path, "coco", "val")):
        os.makedirs(os.path.join(base_path, "coco", "val"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name",default='book', help="class name", type=str)
    parser.add_argument("--input", default='dataset/coco_500_extract',help="json file path (labelme)", type=str)
    parser.add_argument("--output",default='dataset/coco/coco_700_extracted_fourpoints', help="output file path (coco format)", type=str)
    parser.add_argument("--join_num", help="number of join", type=int, default=1)
    parser.add_argument("--ratio", help="train and test split ratio", type=float, default=0.2)
    args = parser.parse_args()

    labelme_path = args.input
    saved_coco_path = args.output

    init_dir(saved_coco_path)  # 初始化COCO数据集的文件夹结构

    json_list_path = glob.glob(labelme_path + "/*.json")
    train_path, val_path = train_test_split(json_list_path, test_size=args.ratio)
    print('{} for training'.format(len(train_path)),
          '\n{} for testing'.format(len(val_path)))
    print('Start transform please wait ...')

    l2c_train = Labelme2coco_keypoints(args)  # 构造数据集生成类

    # 生成训练集
    train_keypoints = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_keypoints, os.path.join(saved_coco_path, "coco", "annotations", "keypoints_train.json"))

    # 生成验证集
    l2c_val = Labelme2coco_keypoints(args)
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, os.path.join(saved_coco_path, "coco", "annotations", "keypoints_val.json"))

    # 拷贝 labelme 的原始图片到训练集和验证集里面
    for file in train_path:
        shutil.copy(file.replace("json", "jpg"), os.path.join(saved_coco_path, "coco", "train"))
    for file in val_path:
        shutil.copy(file.replace("json", "jpg"), os.path.join(saved_coco_path, "coco", "val"))
