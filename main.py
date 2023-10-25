from ultralytics import YOLO
import sys
# sys.path.append('D:/yolo_new/ultralytics')
if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/v8/yolov8_colforme_ef.yaml").load('./models/yolov8n-seg.pt')
    model.train(**{'cfg':'ultralytics/cfg/default.yaml'})

# from ultralytics import YOLO
#
# Create a new YOLO model from scratch
# model = YOLO('yolov8.yaml')
#
# # Load a pretrained YOLO model (recommended for training)
# model = YOLO('models/yolov8l-seg.pt')

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
# model.train(**{'cfg':'ultralytics/cfg/default.yaml'})
#
# # Evaluate the model's performance on the validation set
# results = model.val()
#
# # Perform object detection on an image using the model
# results = model('')
#
# # Export the model to ONNX format
# success = model.export(format='pt')