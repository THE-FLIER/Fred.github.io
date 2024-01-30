import numpy as np
import cv2
import glob
import os
import json
import math
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

def sort_keypoints(keypoints):
    # 对每个四边形，取其所有点的x坐标的平均值作为排序依据
    sorted_keypoints = sorted(keypoints, key=lambda quad: np.mean([point[0] for point in quad]))
    return np.array(sorted_keypoints)

def main():
    book_path = 'dataset/augment_1_22'
    shelve_path = glob.glob(book_path + "/*.json")
    ori_book_image_save_path_ = 'dataset/lishui_crop'
    for jsonfile in shelve_path:
        a = 1
        b = 1
        pure_name = os.path.basename(jsonfile)
        pure_name = pure_name[:-5]

        # json
        book_spine_path = os.path.join(book_path, f'{pure_name}.json')
        images = os.path.join(book_path, f'{pure_name}.jpg')


        f1_book = open(book_spine_path, 'rb')
        book_info = json.load(f1_book)

        ori_height = book_info['imageHeight']
        ori_width = book_info['imageWidth']

        book_list = []

        for point_json in book_info["shapes"]:
            np_points = np.array(point_json["points"], np.int32)
            if len(np_points) ==4:
                book_list.append(np_points)
        book_list =  sort_keypoints(book_list)
        image = cv2.imread(images)
        os.makedirs(f"{ori_book_image_save_path_}/{pure_name}",exist_ok=True)

        for ori_books in book_list:

            points = np.int32(ori_books)
            crop = crop_rec(points, image)
            # 保存
            cv2.imwrite(f"{ori_book_image_save_path_}/{pure_name}/{b}.jpg", crop)
            b += 1

        print(f'{pure_name} Finished')

main()