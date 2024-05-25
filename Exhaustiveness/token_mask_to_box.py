import matplotlib.pyplot as plt
from PIL import Image
import imgviz
import shutil
import argparse
import os
import tqdm
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm

def get_largest_cc_box(mask: np.array):
    from skimage.measure import label as measure_label
    labels = measure_label(mask)  # get connected components
    largest_cc_index = np.argmax(np.bincount(labels.flat)[1:]) + 1
    mask = np.where(labels == largest_cc_index)
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1
    return [xmin, ymin, xmax, ymax]

def save_box_list(box_list, fsave, img_path, save_info=True):
    for i in range(len(box_list)):
        info = img_path + ">" + str(i).rjust(4, '0') + " " + " ".join(str(num) for num in box_list[i]) + "\n"
        if save_info:
            fsave.write(info)
        else:
            print(info)


dst_root = "/home/ybmiao/output/Exhaust/tokencut/voc"
img_root = "/media/media01/qysun/data/VOC/VOC07_test/VOCdevkit/VOC2007/JPEGImages/"
mask_root = os.path.join(dst_root, "mask")
# mask_root = "/home/ybmiao/output/test/VOC07_trainval"
dst_f = open(os.path.join(dst_root,"box_ref.txt"),"w")
for path in tqdm(os.listdir(mask_root)):
    box_list = []
    file_path = os.path.join(mask_root,path)
    img_path = os.path.join(img_root,path)
    mask = np.load(file_path)
    box = get_largest_cc_box(mask)

    P = 8
    xmin, ymin, xmax, ymax = get_largest_cc_box(mask)
    box_up_sampled = [xmin * P, ymin * P, xmax * (P + 1), ymax * (P + 1)]
    # print(box_up_sampled)
    box_list.append(box_up_sampled)
    save_box_list(box_list,dst_f,img_path)