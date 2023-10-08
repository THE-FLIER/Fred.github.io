import json
import cv2
import numpy as np
import requests
import base64
from PIL import Image
import io
import  os
def base64_to_pil(image):
    img = Image.open(io.BytesIO(base64.b64decode(image)))
    return img
#请求
def pre(img):

    files = {'file': open(image, 'rb').read()}
    url = 'http://localhost:5000/predict'
    response = requests.post(url, files=files)
    output = response.json()["content"]

    return output

#返回数据处理
def get_crop(output):
    img_list = []
    for i in output:
        outputs = base64_to_pil(i)
        outputs = np.asarray(outputs)
        img_list.append(outputs)
    return img_list

#保存图片
def run(save_path, images):
    name, ext = os.path.splitext(os.path.basename(images))

    pre_out = pre(images)
    crops = get_crop(pre_out)
    index = 1
    for i in crops:
        image = i
        cv2.imwrite(save_path+f'{name}_{index}.jpg', image)
        index += 1

if __name__ == '__main__':
    image = "test_pics/total/1/1_1/1_1_1.jpg"
    save_path = "test_pics/crops/"
    run(save_path, image)