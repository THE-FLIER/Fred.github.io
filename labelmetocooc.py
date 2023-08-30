#lambelme  标 转  yolov
import json
def lambelme_json_label_to_yolov_seg_label():
    import glob
    import numpy as np
    json_path = "test_pics/test2/5/1/test2017/";
    json_files = glob.glob(json_path + "/*.json")
    for json_file in json_files:
        # if json_file != r"C:\Users\jianming_ge\Desktop\code\handle_dataset\water_street\223.json":
        #     continue
        print(json_file)
        f = open(json_file)
        json_info = json.load(f)
        # print(json_info.keys())
        #img = cv2.imread(os.path.join(json_path, json_info["imagePath"]))
        height=1440
        width=2560 
        np_w_h = np.array([[width, height]], np.int32)
        txt_file = json_file.replace(".json", ".txt")
        f = open(txt_file, "a")
        for point_json in json_info["shapes"]:
            txt_content = ""
            np_points = np.array(point_json["points"], np.int32)
            norm_points = np_points / np_w_h
            norm_points_list = norm_points.tolist()
            txt_content += "0 " + " ".join([" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
            f.write(txt_content)

lambelme_json_label_to_yolov_seg_label()