from PIL import Image
from ultralytics import YOLO
import os
input = 'datasets/bookshelf-point/images/val'
output = 'results/point/'

# Load a pretrained YOLOv8n model
model = YOLO('models/bookshelf-point(0.98).pt')
for file_name in os.listdir(input):
    # Run inference on 'bus.jpg'
    source_path = os.path.join(input, file_name)
    results = model(source_path)  # results list
    a = file_name[:-5]
    # Show the results
    for r in results:
        im_array = r.plot(boxes=False, labels=False)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # im.show()  # show image
        im.save(f'{output}{file_name}')