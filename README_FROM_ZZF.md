#训练脚本

--main.py

#推理脚本 

--test_model.py

#labelme转成训练数据集

labelmetococo.py

#切割labelme json文件图片

poly_trans.py


#关于服务部署

1.python app.py

2.hub.py 为请求客户端，需输入图片以及保存路径。若需批次处理，需自行编写。