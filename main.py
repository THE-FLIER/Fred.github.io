from ultralytics import YOLO
import sys
# sys.path.append('D:/yolo_new/ultralytics')
if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/v8/yolov8-seg-p6-cbam.yaml").load('./models/yolov8l-seg.pt')
    model.train(**{'cfg': 'ultralytics/cfg/default.yaml'})

