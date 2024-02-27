from ultralytics import YOLO
import sys
# sys.path.append('D:/yolo_new/ultralytics')
if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/v8/yolov8m-point-strpooling.yaml").load('./models/yolov8m-pose.pt')
    model.train(**{'cfg': 'ultralytics/cfg/default-point.yaml'})


