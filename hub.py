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

def is_asc(asc):
    try:
        asc.isascii()
        return True
    except AttributeError:
        return False

#
def get_crop(output):
    img_list = []
    if 'NONE' not in output:
        for i in output:
            if is_asc(i):
                outputs = base64_to_pil(i)
                outputs = np.asarray(outputs)
                img_list.append(outputs)
            else:
                img_list = output
    else:
        img_list = False

    return img_list

#保存图片
def run(args):
    save_path = args.infer_results
    images = args.image_path
    os.makedirs(save_path, exist_ok=True)
    for i in os.listdir(images):
        if i.endswith('.jpg') or i.endswith('.png') :
            #参数
            confidence = args.conf
            coordinate = None
            parament = {'conf': confidence, 'coordinate': coordinate}
            img_path = os.path.join(images, i)
            name, ext = os.path.splitext(os.path.basename(img_path))
            pre_out = pre(img_path, parament)
            crops = get_crop(pre_out)

            index = 1
            if crops:
                for i in crops:
                    crop_path = save_path+f'{name}_{index}.jpg'
                    i_rgb = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(crop_path, i_rgb)
                    index += 1
            else:
                print(f'Images predict:{crops}')

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
        "--image_path", type=str, default="test_pics/1/shelf", help="Path Of Image To Infer"
    )
    parser.add_argument(
        "--infer_results", type=str, default="test_pics/crops/", help="Path Of Infer_Crops To Save"
    )
    parser.add_argument(
        "--conf", type=str, default='0.5', help="Confidence Of Predict"
    )
    args = parser.parse_args()
    run(args)