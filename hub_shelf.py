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

def predict(img, paraments):
    parament_binary = pickle.dumps(paraments)
    files = {'file': open(img, 'rb').read(), 'parament': parament_binary}
    url = args.address
    response = requests.post(url, files=files)
    output = response.json()["content"]

    return output

#保存
def run(args):
    save_path = args.infer_results
    images = args.image_path
    confidence = args.conf
    parament = {'conf': confidence}
    os.makedirs(save_path, exist_ok=True)
    for i in os.listdir(images):
        if i.endswith('.jpg') or i.endswith('.png') :
            img_path = os.path.join(images, i)
            pre_out = predict(img_path, parament)
            points = json.loads(pre_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--infer_results", type=str, default="test_pics/crops/", help="Path Of Infer_Crops To Save"
    )
    parser.add_argument(
        "--image_path", type=str, default="test_pics/1/shelf", help="Path Of Image To Infer"
    )
    parser.add_argument(
        "--conf", type=str, default='0.85', help="Confidence Of Predict"
    )
    parser.add_argument(
        "--address", type=str, default='http://172.16.1.152:5000/predict', help="Api Address"
    )
    args = parser.parse_args()
    run(args)
