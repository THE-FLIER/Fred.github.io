import json
import cv2
import numpy as np
import requests
import base64
from PIL import Image
import io
import  os
import argparse
import pickle
def base64_to_pil(image):
    img = Image.open(io.BytesIO(base64.b64decode(image)))
    return img
#请求

#返回数据处理
def get_crop(output):
    img_list = []
    for i in output:
        outputs = base64_to_pil(i)
        outputs = np.asarray(outputs)
        img_list.append(outputs)
    return img_list

#保存图片
def run(args):
    save_path = args.infer_results
    images = args.image_path
    confidence = args.conf
    parament = {'conf': confidence}
    os.makedirs(save_path, exist_ok=True)
    for i in os.listdir(images):
        if i.endswith('.jpg') or i.endswith('.png') :
            img_path = os.path.join(images, i)
            name, ext = os.path.splitext(os.path.basename(img_path))
            pre_out = pre(img_path, parament)
            crops = get_crop(pre_out)
            index = 1
            for i in crops:
                crop_path = save_path+f'{name}_{index}.jpg'
                cv2.imwrite(crop_path, i)
                index += 1

def pre(img, paraments):
    parament_binary = pickle.dumps(paraments)
    files = {'file': open(img, 'rb').read(), 'parament': parament_binary}
    url = 'http://172.16.1.152:5000/predict'
    response = requests.post(url, files=files)
    output = response.json()["content"]

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--image_path", type=str, default="test_pics/1", help="Path Of Image To Infer"
    )
    parser.add_argument(
        "--infer_results", type=str, default="test_pics/crops/", help="Path Of Infer_Crops To Save"
    )
    parser.add_argument(
        "--conf", type=str, default='0.7', help="Confidence Of Predict"
    )
    args = parser.parse_args()
    run(args)