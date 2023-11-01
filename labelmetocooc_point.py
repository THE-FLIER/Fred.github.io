#lambelme  标 转  yolov
import json
import os
'''
会在同一目录下生成txt训练文件
'''
def lambelme_json_label_to_yolov_seg_label(json_path):
    import glob
    import numpy as np
    json_path = json_path
    json_files = glob.glob(json_path + "/*.json")
    for json_file in json_files:
        # if json_file != r"C:\Users\jianming_ge\Desktop\code\handle_dataset\water_street\223.json":
        #     continue
        print(json_file)
        f = open(json_file)
        json_info = json.load(f)
        # print(json_info.keys())
        #img = cv2.imread(os.path.join(json_path, json_info["imagePath"]))
        height = 1440
        width = 2560
        np_w_h = np.array([[width, height]], np.int32)
        txt_file = os.path.basename(json_file)
        txt_file = f'datasets/bookshelf-point/labels/train/{txt_file[:-5]}.txt'
        f = open(txt_file, "w")
        for point_json in json_info["shapes"]:
            txt_con = ""
            np_points = np.array(point_json["points"], np.int32)

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

            txt_con += f'0 {center_x} {center_y} {width_} {height_} '
            norm_points = np_points / np_w_h
            norm_points_list = norm_points.tolist()
            txt_content = f"{txt_con}" + " ".join([" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
            f.write(txt_content)


if __name__=="__main__":
    json_path = "datasets/bookshelf-point/test/"
    lambelme_json_label_to_yolov_seg_label(json_path)