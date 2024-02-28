# 目前训练最好的书脊和书格检测模型
### 书脊
    yolov8/hub_service_book/best_book.pt
    项目结果路径：/home/zf/yolov8/runs/pose/1_15_shelves_13_points
        
### 书格
    yolov8/hub_service_shelf/best_shelf.pt

# 训练模型流程
    1)main.py 中修改预训练模型，模型参数配置文件，训练参数配置
    2)可选模型参数从小到达：n,s,m,l,x
    3)模型参数配置文件：yolov8/ultralytics/cfg/models/v8目录下，
                     配置文件要根据选择模型大小调配，如选了yolov8n,
                     则也要选择yolov8n-point.yaml.
    4)训练参数配置：yolov8/ultralytics/cfg/default-point.yaml，
                 需要设置的有：1.预训练模型2.数据集参数3.其他训练参数及超参数。
                 1.预训练模型：自行官网下载，确保任务正确
                 2.数据集参数：yolov8/ultralytics/cfg/datasets/book-multi_point.yaml
                            设置根目录，以及类别。如果是关键点,则关键点为[x,2].
                 3.训练参数及超参数：目前不需要调配。
    调配好后启动训练。

# 数据处理及生成
    1)总数居批量处理(labelme转yolo)：yolov8/persptrans.py
                   数据按照规定标注，包含书脊，书格.对该数据进行处理，获得书脊，书格标准训练数据。
                   labelme透视变换图片生成一系列训练文件。
                   选择好标注数据，分配保存路径即可获得。 
    2)关键点检测labelme转yolo格式：yolov8/labelme2yolo
    3)关键点检测labelme转coco格式（mmpose）:yolov8/labelmetopose.py
    4)切割labelme json文件图片:poly_trans.py

# 推理脚本
### - ``
    1)yolov8/predict.py：本脚本可以获得分割图片，分割掩模大图。
    2)yolov8/predict4labelme.py:推理后可以获得labelme文件，查看效果，调整再加入新一轮训练数据中。
     
# 服务及docker
## 关于服务测试
1. 运行 `app_request.py` 启动服务
2. `hub.py` 为请求客户端样例，传入图片和confidence。
## 关于docker
    分为shelf，book关键点检测两个docker，文件夹都包含pt模型文件，yolo项目文件，以及dockerfile。
    docker build,docker run即可,需开启gpu。
    1.yolov8/hub_service_shelf
    2.yolov8/hub_service_book

