"""
get semantic segmentation annotations from coco data set.
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imgviz
import argparse
import os
import tqdm
from pycocotools.coco import COCO
from torchvision import transforms as pth_transforms
import cv2
import xml.etree.ElementTree as ET


def get_names_and_bboxs(xml_file_path):
    # 解析XML文件
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    info_list = []
    # 遍历XML中的所有元素
    for obj in root.iter('object'):
        # 获取object中的name元素
        name = obj.find('name').text
        # 获取object中的bndbox元素的坐标
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        info_list.append((name, (xmin, ymin, xmax, ymax)))

    return info_list


def mask_iou(mask1:np.array, mask2:np.array):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    # 计算 IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou

def box_iou(box1, box2):
    """
    :param box1: (x1, y1, x2, y2)
    :param box2: (x1, y1, x2, y2)
    :return:
    """
    # 计算两个box的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算相交矩形的坐标
    left = max(box1[0], box2[0])
    right = min(box1[2], box2[2])
    top = max(box1[1], box2[1])
    bottom = min(box1[3], box2[3])

    # 计算相交矩形的面积
    inter_area = max(0, right - left) * max(0, bottom - top)

    # 计算IoU
    iou = inter_area / (area1 + area2 - inter_area)

    return iou



def cal_recall(num, ann_sum, thre, ref_list, gt_list):
    for i in range(len(gt_list)):
        for j in range(len(ref_list)):
            _, box = gt_list[i]
            iou_score = box_iou(box, ref_list[j])
            if iou_score > thre:
                # print(iou_score)
                ann_sum += 1
                break
    return ann_sum

def main():
    annotation_file = "/home/ybmiao/data/VOC/VOC07_test/VOCdevkit/VOC2007/Annotations"
    img_root = "/home/ybmiao/data/VOC/VOC07_test/VOCdevkit/VOC2007/JPEGImages"
    file_name_path = "/media/media01/qysun/data/VOC/VOC2007_test_filename.txt"
    # mask_root = "/media/media01/ybmiao/output/kcut/1_10/VOC_test2007/mask"
    # box_path = "/media/media01/ybmiao/output/kcut/1_10/VOC_test2007/box_ref.txt"
    # box_path = "/home/ybmiao/output/Exhaust/cutler/voc/box_ref.txt"
    # box_path = "/home/ybmiao/output/Exhaust/tokencut/voc/box_ref.txt"
    box_path = "/media/media01/qysun/results/ksum/temp/voc/box_ref.txt"
    # box_path = "/home/ybmiao/output/Exhaust/dss/voc_test2007/box_ref.txt"
    # mask_root = "/home/ybmiao/output/Exhaust/dss/voc_test2007/mask"
    # mask_root = "/home/ybmiao/output/Exhaust/cutler/voc/mask"
    # mask_root = "/home/ybmiao/output/Exhaust/tokencut/voc/mask"


    # 读取voc数据集标注xml文件
    file_names = open(file_name_path,"r").readlines()
    print("file_names len:{}".format(len(file_names)))

    all_ann = 0
    miou10_ann = 0
    miou20_ann = 0
    miou30_ann = 0
    miou40_ann = 0
    miou50_ann = 0
    miou60_ann = 0
    miou70_ann = 0
    miou80_ann = 0
    miou90_ann = 0

    box_info_dict = {}
    with open(box_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            img_name = line[0].split(">")[0].split("/")[-1].split(".")[0] + ".jpg"
            box = list(map(int, line[1:]))
            if img_name not in box_info_dict:
                box_info_dict[img_name] = []
            box_info_dict[img_name].append(box)
    # print(box_info_dict["006127.jpg"])
    # print("box_info_dict len:{}".format(len(box_info_dict)))
    # exit()

    for idx, file_name in tqdm.tqdm(enumerate(file_names)):
        file_name = file_name.strip().split(".")[0]
        xml_file_path = os.path.join(annotation_file, file_name + ".xml")
        img_path = os.path.join(img_root, file_name + ".jpg")
        info_list = get_names_and_bboxs(xml_file_path)
        all_ann += len(info_list)
        if file_name + ".jpg" not in box_info_dict:
            continue
        if len(box_info_dict[file_name + ".jpg"]) == 0:
            continue
        # print(file_name)
        # print(info_list)
        # exit()
        miou10_ann = cal_recall(len(info_list), miou10_ann, 0.1, box_info_dict[file_name + ".jpg"], info_list)
        miou20_ann = cal_recall(len(info_list), miou20_ann, 0.2, box_info_dict[file_name + ".jpg"], info_list)
        miou30_ann = cal_recall(len(info_list), miou30_ann, 0.3, box_info_dict[file_name + ".jpg"], info_list)
        miou40_ann = cal_recall(len(info_list), miou40_ann, 0.4, box_info_dict[file_name + ".jpg"], info_list)
        miou50_ann = cal_recall(len(info_list), miou50_ann, 0.5, box_info_dict[file_name + ".jpg"], info_list)
        miou60_ann = cal_recall(len(info_list), miou60_ann, 0.6, box_info_dict[file_name + ".jpg"], info_list)
        miou70_ann = cal_recall(len(info_list), miou70_ann, 0.7, box_info_dict[file_name + ".jpg"], info_list)
        miou80_ann = cal_recall(len(info_list), miou80_ann, 0.8, box_info_dict[file_name + ".jpg"], info_list)
        miou90_ann = cal_recall(len(info_list), miou90_ann, 0.9, box_info_dict[file_name + ".jpg"], info_list)

    # print("miou10:\n", miou10_ann, miou10_ann / all_ann)
    # print("miou20:\n", miou20_ann, miou20_ann / all_ann)
    # print("miou30:\n", miou30_ann, miou30_ann / all_ann)
    # print("miou40:\n", miou40_ann, miou40_ann / all_ann)
    # print("miou50:\n", miou50_ann, miou50_ann / all_ann)
    # print("miou60:\n", miou60_ann, miou60_ann / all_ann)
    # print("miou70:\n", miou70_ann, miou70_ann / all_ann)
    # print("miou80:\n", miou80_ann, miou80_ann / all_ann)
    # print("miou90:\n", miou90_ann, miou90_ann / all_ann)

    print("10.0,0.1,{}".format(miou10_ann / all_ann))
    print("20.0,0.2,{}".format(miou20_ann / all_ann))
    print("30.0,0.3,{}".format(miou30_ann / all_ann))
    print("40.0,0.4,{}".format(miou40_ann / all_ann))
    print("50.0,0.5,{}".format(miou50_ann / all_ann))
    print("60.0,0.6,{}".format(miou60_ann / all_ann))
    print("70.0,0.7,{}".format(miou70_ann / all_ann))
    print("80.0,0.8,{}".format(miou80_ann / all_ann))
    print("90.0,0.9,{}".format(miou90_ann / all_ann))

    print(all_ann)



if __name__ == '__main__':
    main()
