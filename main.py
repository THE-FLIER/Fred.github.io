from ultralytics import YOLO
import sys
# sys.path.append('D:/yolo_new/ultralytics')
if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/v8/yolov8-pose-p6-cbam.yaml").load('./models/yolov8x-pose-p6.pt')
    model.train(**{'cfg': 'ultralytics/cfg/default-point.yaml'})

