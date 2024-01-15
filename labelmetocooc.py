#lambelme  标 转  yolov
import json
import os
'''
会在同一目录下生成txt训练文件
'''

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def lambelme_json_label_to_yolov_seg_label(json_path):
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
        # print(json_info.keys())
        #img = cv2.imread(os.path.join(json_path, json_info["imagePath"]))
        height = json_info['imageHeight']
        width = json_info['imageWidth']
        np_w_h = np.array([[width, height]], np.int32)
        txt_file = os.path.basename(json_file)
        txt_file = f'datasets/bookandshelf-point/labels/train/{txt_file[:-5]}.txt'
        f = open(txt_file, "w")
        for point_json in json_info["shapes"]:
            txt_content = ""
            if is_number(point_json["label"]):
                np_points = np.array(point_json["points"], np.int32)
                norm_points = np_points / np_w_h
                norm_points_list = norm_points.tolist()
                if float(point_json["label"]) == 4.0:
                    txt_content += "3 " + " ".join([" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
                    f.write(txt_content)
                elif float(point_json["label"]) == 0.0:
                    txt_content += "0 " + " ".join(
                        [" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
                    f.write(txt_content)
                elif float(point_json["label"]) == 1.0:
                    txt_content += "1 " + " ".join(
                        [" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
                    f.write(txt_content)
                elif float(point_json["label"]) == 2.0:
                    txt_content += "2 " + " ".join(
                        [" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
                    f.write(txt_content)




if __name__=="__main__":
    json_path = "datasets/first_batch"
    lambelme_json_label_to_yolov_seg_label(json_path)