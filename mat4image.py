import json
import os

import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO
import glob
def getImgFromJson(filename):
    # 读取并解码json文件
    with open(filename, 'r') as f:
        jsonData = json.load(f)

    # 解码图像数据
    im_v = base64.b64decode(jsonData['imageData'])

    # 使用PIL读取图像数据流
    img = Image.open(BytesIO(im_v))

    # 获取图像的宽度和高度
    w, h = img.size

    # 将图像数据转换为numpy数组，并调整形状
    p = np.array(img.getdata()).reshape(h, w, 3)

    # 调整颜色通道的顺序
    img = np.concatenate((p[:,:,2:3], p[:,:,1:2], p[:,:,0:1]), axis=2)
    book_shapes = []
    for point_json in jsonData["shapes"]:
        np_points = np.array(point_json["points"], np.int32)
        book_shapes.append({"label": "1", "points": np_points.tolist()})
    pure_name = os.path.basename(filename)
    pure_name = pure_name[:-5]
    book_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": book_shapes,
        "imagePath": f'{pure_name}.jpg',
        "imageData": None,
        "imageWidth": w,
        "imageHeight": h
    }

    return img,book_data

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

def main():
    json_path = 'dataset/661_test/dataset_661'
    write_path = 'dataset/661_test/images'
    os.makedirs(write_path,exist_ok=True)
    json_files = glob.glob(json_path + "/*.json")

    for json_file in json_files:
        txt_file = os.path.basename(json_file)
        pure_name = txt_file[:-5]
        image,book_data = getImgFromJson(json_file)
        cv2.imwrite(f'{write_path}/{txt_file[:-5]}.jpg',image)
        write_to_json(write_path, f'{pure_name}', book_data)
        print(f'{txt_file} transferred')
main()