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

    outputs = []
    for i in image:
        img = np.asarray(Image.open(io.BytesIO(base64.b64decode(i))))
        outputs.append(img)

    return outputs

def is_asc(asc):
    try:
        asc.isascii()
        return True
    except AttributeError:
        return False

#
def get_crop(output):
    img_list = []

    for i in output:

        outputs = base64_to_pil(i)

        img_list.append(outputs)

    return img_list

#保存图片
def run(args):
    save_path = args.infer_results
    images = args.image_path
    os.makedirs(save_path, exist_ok=True)
    img_list = []
    for i in os.listdir(images):
        if i.endswith('.jpg') or i.endswith('.png'):
            img_path = os.path.join(images, i)
            #参数
            img_list.append(cv2.imread(img_path))

    confidence = args.conf
    coordinate = None
    parament = {'conf': confidence, 'coordinate': coordinate}

    pre_out = pre(img_list, parament)

    crops = get_crop(pre_out)

    index = 1
    name = 1

    if crops:
        for i in crops:
            for j in i:
                crop_path = save_path + f'{name}_{index}.jpg'
                cv2.imwrite(crop_path, j)
                index += 1
            name += 1
        else:
                print(f'Images predict:{crops}')


def pre(img, paraments):
    parament_binary = pickle.dumps(paraments)
    imgs_bytes = pickle.dumps(img)
    files = {'file': imgs_bytes, 'parament': parament_binary}
    url = 'http://172.16.1.152:5003/predict'
    response = requests.post(url, files=files)
    output = response.json()["content"]

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--image_path", type=str, default="dataset/new", help="Path Of Image To Infer"
    )
    parser.add_argument(
        "--infer_results", type=str, default="test_pics/1_29/crops/", help="Path Of Infer_Crops To Save"
    )
    parser.add_argument(
        "--conf", type=str, default='0.5', help="Confidence Of Predict"
    )
    args = parser.parse_args()
    run(args)