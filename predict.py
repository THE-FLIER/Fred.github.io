from PIL import Image
from ultralytics import YOLO
import os
import cv2
import numpy as np
input = 'dataset/bookshelf-point(4)/images/val'

output = 'results/point1/'

# Load a pretrained YOLOv8n model
model = YOLO('models/best.pt')
def polygons_to_mask2(img_shape, polygons):
    '''
    边界点生成mask
    :param img_shape: [h,w]
    :param polygons: labelme JSON中的边界点格式 [[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]
    :return:
    '''
    mask = np.zeros(img_shape, dtype=np.uint8)
    polygons = np.asarray(polygons, np.int32) # 这里必须是int32，其他类型使用fillPoly会报错
    # cv2.fillPoly(mask, polygons, 1) # 非int32 会报错
    cv2.fillConvexPoly(mask, polygons, 1)  # 非int32 会报错

    return mask

for file_name in os.listdir(input):
    # Run inference on 'bus.jpg'
    source_path = os.path.join(input, file_name)
    ori_img = cv2.imread(source_path)
    h, w, _ = ori_img.shape
    results = model(source_path, conf=0.9, imgsz=1280, save_txt=False, save_crop=False, boxes=False, device='0')  # results list
    a = file_name[:-5]
    # Show the results
    for r in results:
        # im_array = r.plot(boxes=False, labels=False)  # plot a BGR numpy array of predictions
        # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # # im.show()  # show image
        # im.save(f'{output}{file_name}')
        keypoints = r.keypoints.xyn.cpu().numpy()
        c = 1
        black_img = np.zeros([h, w], dtype=np.uint8)
        res = np.zeros_like(ori_img)

        for points in keypoints:
            if len(points) != 0:
                points[:, 0] = points[:, 0] * w
                points[:, 1] = points[:, 1] * h

                mask = polygons_to_mask2([h, w], points)

                rect = cv2.boundingRect(points)

                # 使用边界框截取矩形区域
                res[mask > 0] = ori_img[mask > 0]

                x = points[:, 0]
                y = points[:, 1]
                y1 = int(min(y))
                y2 = int(max(y))
                x1 = int(min(x))
                x2 = int(max(x))

                crop_img = res[y1:y2, x1:x2]

                # 保存结果
                # cv2.imwrite(f'./results/point-crop/{a}_{c}.jpg', crop_img)
                c += 1

                polygons = np.asarray(points, np.int32)
                cv2.fillConvexPoly(black_img, polygons, 1)
                cv2.fillConvexPoly(black_img, polygons, 1)

        cv2.imwrite(f"./results/point1/crop/{a}.jpg", black_img * 255)
        cv2.imwrite(f"./results/point1/ori/{a}_ori.jpg", res)