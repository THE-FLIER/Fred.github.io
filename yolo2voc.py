from __future__ import print_function
import  os, cv2, shutil
# from lxml import etree, objectify
# from tqdm import tqdm
# from PIL import Image
# import numpy as np
# import time
# import os.path as osp
# import json

# def cover_copy(src,dst):
#     '''
#     src和dst都必须是文件，该函数是执行覆盖操作
#     '''
#     if os.path.exists(dst):
#         os.remove(dst)
#         shutil.copy(src,dst)
#     else:
#         shutil.copy(src,dst)
#
# def yolo2voc(sourcedir,savedir,class_names):
#     """_summary_
#
#     Args:
#         sourcedir (_type_): /yournewpath 写到images 和labels上一级
#         savedir (_type_): 写到JPEGImages和Annotations上一级
#         class_names(list): yolo数据中所有的类别，顺序要与索引对应上
#     """
#
#     img_savepath= osp.join(savedir,'JPEGImages')
#     ann_savepath=osp.join(savedir,'Annotations')
#     for p in [img_savepath,ann_savepath]:
#         if osp.exists(p):
#             shutil.rmtree(p)
#             os.makedirs(p)
#         else:
#             os.makedirs(p)
#     filenames = os.listdir(osp.join(sourcedir,'images'))
#     for filename in tqdm(filenames):
#         filepath = osp.join(sourcedir,'images',filename)
#         annfilepath=osp.join(sourcedir,'labels',osp.splitext(filename)[0]+'.txt')
#         annopath = osp.join(ann_savepath,osp.splitext(filename)[0] + ".xml") #生成的xml文件保存路径
#         dst_path = osp.join(img_savepath,filename)
#         im = Image.open(filepath)
#         image = np.array(im).astype(np.uint8)
#         w=image.shape[1]
#         h=image.shape[0]
#         cover_copy(filepath, dst_path)#把原始图像复制到目标文件夹
#         anns=[i.strip().split() for i in open(annfilepath).readlines()]
#         objs = []
#         for ann in anns:
#             name = class_names[int(ann[0])]
#             xcenter = float(ann[1])*w
#             ycenter = float(ann[2])*h
#             bw=float(ann[3])*w
#             bh=float(ann[4])*h
#             xmin = (int)(xcenter-bw/2)
#             ymin = (int)(ycenter-bh/2)
#             xmax = (int)(xcenter+bw/2)
#             ymax = (int)(ycenter+bh/2)
#             obj = [name, 1.0, xmin, ymin, xmax, ymax]
#             #标错框在这里
#             if not(xmin-xmax==0 or ymin-ymax==0):
#                 objs.append(obj)
#
#         E = objectify.ElementMaker(annotate=False)
#         anno_tree = E.annotation(
#             E.folder('VOC'),
#             E.filename(filename),
#             E.source(
#                 E.database('YOLO'),
#                 E.annotation('VOC'),
#                 E.image('YOLO')
#             ),
#             E.size(
#                 E.width(image.shape[1]),
#                 E.height(image.shape[0]),
#                 E.depth(image.shape[2])
#             ),
#             E.segmented(0)
#         )
#
#         for obj in objs:
#             E2 = objectify.ElementMaker(annotate=False)
#             anno_tree2 = E2.object(
#                 E.name(obj[0]),
#                 E.pose(),
#                 E.truncated("0"),
#                 E.difficult(0),
#                 E.bndbox(
#                     E.xmin(obj[2]),
#                     E.ymin(obj[3]),
#                     E.xmax(obj[4]),
#                     E.ymax(obj[5])
#                 )
#             )
#             anno_tree.append(anno_tree2)
#         etree.ElementTree(anno_tree).write(annopath, pretty_print=True)
# yolo2voc('dataset/661_seg','dataset/voc_661',['book'])


import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np

import labelme


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClass"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClassPNG"))
    if not args.noviz:
        os.makedirs(
            osp.join(args.output_dir, "SegmentationClassVisualization")
        )
    os.makedirs(osp.join(args.output_dir, "SegmentationObject"))
    os.makedirs(osp.join(args.output_dir, "SegmentationObjectPNG"))
    if not args.noviz:
        os.makedirs(
            osp.join(args.output_dir, "SegmentationObjectVisualization")
        )
    print("Creating dataset:", args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        out_cls_file = osp.join(
            args.output_dir, "SegmentationClass", base + ".npy"
        )
        out_clsp_file = osp.join(
            args.output_dir, "SegmentationClassPNG", base + ".png"
        )
        if not args.noviz:
            out_clsv_file = osp.join(
                args.output_dir,
                "SegmentationClassVisualization",
                base + ".jpg",
            )
        out_ins_file = osp.join(
            args.output_dir, "SegmentationObject", base + ".npy"
        )
        out_insp_file = osp.join(
            args.output_dir, "SegmentationObjectPNG", base + ".png"
        )
        if not args.noviz:
            out_insv_file = osp.join(
                args.output_dir,
                "SegmentationObjectVisualization",
                base + ".jpg",
            )

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        cls, ins = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        ins[cls == -1] = 0  # ignore it.

        # class label
        labelme.utils.lblsave(out_clsp_file, cls)
        np.save(out_cls_file, cls)
        if not args.noviz:
            clsv = imgviz.label2rgb(
                cls,
                imgviz.rgb2gray(img),
                label_names=class_names,
                font_size=15,
                loc="rb",
            )
            imgviz.io.imsave(out_clsv_file, clsv)

        # instance label
        labelme.utils.lblsave(out_insp_file, ins)
        np.save(out_ins_file, ins)
        if not args.noviz:
            instance_ids = np.unique(ins)
            instance_names = [str(i) for i in range(max(instance_ids) + 1)]
            insv = imgviz.label2rgb(
                ins,
                imgviz.rgb2gray(img),
                label_names=instance_names,
                font_size=15,
                loc="rb",
            )
            imgviz.io.imsave(out_insv_file, insv)

if __name__ == "__main__":
    #main()
    import mmcv
    import os
    import os.path as osp
    import random

    """1.检查图像维度"""
    import numpy as np
    from PIL import Image, ImageOps
    from torchvision import transforms

    def get_Image_dim_len(png_dir: str, jpg_dir: str):
        png = Image.open(png_dir)
        png_w, png_h = png.width, png.height
        # 若第十行报错，说明jpg图片没有对应的png图片
        png_dim_len = len(np.array(png).shape)
        assert png_dim_len == 2, "提示:存在三维掩码图"
        jpg = Image.open(jpg_dir)
        jpg = ImageOps.exif_transpose(jpg)
        jpg.save(jpg_dir)
        jpg_w, jpg_h = jpg.width, jpg.height
        print(jpg_w, jpg_h, png_w, png_h)
        assert png_w == jpg_w and png_h == jpg_h, print("提示：%s mask图与原图宽高参数不一致" % (png_dir))


    """2.读取单个图像均值和方差"""


    def pixel_operation(image_path: str):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        means, dev = cv2.meanStdDev(img)
        return means, dev


    """3.分割数据集，生成label文件"""
    # 原始数据集 ann上一级
    data_root = 'dataset/data_dataset_voc'
    # 图像地址
    image_dir = "JPEGImages"
    # ann图像文件夹
    ann_dir = "SegmentationClass"
    # txt文件保存路径
    split_dir = 'ImageSets/Segmentation'
    mmcv.mkdir_or_exist(osp.join(data_root, split_dir))

    png_filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
        osp.join(data_root, ann_dir), suffix='.png')]
    jpg_filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
        osp.join(data_root, image_dir), suffix='.jpg')]
    assert len(jpg_filename_list) == len(png_filename_list), "提示：原图与掩码图数量不统一"
    print("数量检查无误")
    for i in range(10):
        random.shuffle(jpg_filename_list)
    red_num = 0
    black_num = 0
    with open(osp.join(data_root, split_dir, 'trainval.txt'), 'w+') as f:
        length = int(len(jpg_filename_list))
        for line in jpg_filename_list[:length]:
            pngpath = osp.join(data_root, ann_dir, line + '.png')
            jpgpath = osp.join(data_root, image_dir, line + '.jpg')
            get_Image_dim_len(pngpath, jpgpath)
            img = cv2.imread(pngpath, cv2.IMREAD_GRAYSCALE)
            red_num += len(img) * len(img[0]) - len(img[img == 0])
            black_num += len(img[img == 0])
            f.writelines(line + '\n')
        value = red_num / black_num

    train_mean, train_dev = [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]
    with open(osp.join(data_root, split_dir, 'train.txt'), 'w+') as f:
        train_length = int(len(jpg_filename_list) * 7 / 10)
        for line in jpg_filename_list[:train_length]:
            jpgpath = osp.join(data_root, image_dir, line + '.jpg')
            mean, dev = pixel_operation(jpgpath)
            train_mean += mean
            train_dev += dev
            f.writelines(line + '\n')
    with open(osp.join(data_root, split_dir, 'val.txt'), 'w+') as f:
        for line in jpg_filename_list[train_length:]:
            jpgpath = osp.join(data_root, image_dir, line + '.jpg')
            mean, dev = pixel_operation(jpgpath)
            train_mean += mean
            train_dev += dev
            f.writelines(line + '\n')
        train_mean, train_dev = train_mean / length, train_dev / length

    doc = open('均值方差像素比.txt', 'a+')
    doc.write("均值:" + '\n')
    for item in train_mean:
        doc.write(str(item[0]) + '\n')
    doc.write("训练集方差:" + '\n')
    for item in train_dev:
        doc.write(str(item[0]) + '\n')
    doc.write("像素比:" + '\n')
    doc.write(str(value))
    train_mean, train_dev = [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]


