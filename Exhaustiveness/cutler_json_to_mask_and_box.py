import matplotlib.pyplot as plt
from PIL import Image
import imgviz
import shutil
import argparse
import os
import tqdm
from pycocotools.coco import COCO
import numpy as np

def get_largest_cc_box(mask: np.array):
    from skimage.measure import label as measure_label
    labels = measure_label(mask)  # get connected components
    largest_cc_index = np.argmax(np.bincount(labels.flat)[1:]) + 1
    mask = np.where(labels == largest_cc_index)
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1
    return [xmin, ymin, xmax, ymax]

def save_prm_and_box_list(dataset, prm_list, box_list, fsave, mask_save_path, img_path, save_info=True):
    if dataset == "Ins160" or dataset == "Ins335" or dataset == "CUHK-SYSU" or dataset == "gdd":
        name = img_path.split('.')[0].split('/')[-1]
    elif dataset == "instre" or dataset == "test_instre":
        name = "/".join(img_path.split('.')[0].split('/')[-3:]).replace("/", "_")
    elif dataset == "coco" or dataset == "voc":
        name = img_path.split('.')[0].split('/')[-1]
    else:
        raise ValueError("Wrong ref path input!")

    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)

    prm_np = np.array(prm_list).astype(np.uint8)
    if save_info:
        file_name = os.path.join(mask_save_path, name +".npy")
        np.save(file_name, prm_np)
    else:
        print(prm_np)

    for i in range(len(box_list)):
        info = img_path + ">" + str(i).rjust(4, '0') + " " + " ".join(str(num) for num in box_list[i]) + "\n"
        if save_info:
            fsave.write(info)
        else:
            print(info)

# dataset = "Ins160"
dataset = "voc"
# dataset = "instre"
if dataset == "coco":
    dst_root = "/home/ybmiao/output/Exhaust/cutler/coco"
    dst_f = open(os.path.join(dst_root,"box_ref.txt"),"w")
    annotation_file = "/media/media01/ybmiao/output/Exhaust/cutler/annotations/coco_val2017_fixsize480_tau0.15_N3.json"
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        img_path = os.path.join("/home/ybmiao/data/coco/images/val2017",img['file_name'].split("/")[-1])
        if not os.path.exists(img_path):
            raise ValueError("no img")
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        prm_list = []
        box_list = []
        if len(annIds) > 0:
            for i in range(len(anns)):
                mask = coco.annToMask(anns[i])
                prm_list.append(mask)
                box_list.append(get_largest_cc_box(mask))
        save_prm_and_box_list(dataset="coco",prm_list=prm_list,box_list=box_list,fsave=dst_f,mask_save_path=os.path.join(dst_root,"mask"), img_path=img_path)
elif dataset == "voc":
    dst_root = "/home/ybmiao/output/Exhaust/cutler/voc"
    dst_f = open(os.path.join(dst_root,"box_ref.txt"),"w")
    annotation_file = "/home/ybmiao/output/Exhaust/cutler/annotations/imagenet_train_fixsize480_tau0.15_N3.json"
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        img_path = os.path.join("/media/media01/qysun/data/VOC/VOC07_test/VOCdevkit/VOC2007/JPEGImages",img['file_name'].split("/")[-1])
        if not os.path.exists(img_path):
            raise ValueError("no img")
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        prm_list = []
        box_list = []
        if len(annIds) > 0:
            for i in range(len(anns)):
                mask = coco.annToMask(anns[i])
                prm_list.append(mask)
                box_list.append(get_largest_cc_box(mask))
        save_prm_and_box_list(dataset="voc",prm_list=prm_list,box_list=box_list,fsave=dst_f,mask_save_path=os.path.join(dst_root,"mask"), img_path=img_path)
elif dataset == "instre":
    dst_root = "/home/ybmiao/output/Exhaust/cutler/instre"
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    dst_f = open(os.path.join(dst_root,"box_ref.txt"),"a")
    annotation_file = "/media/media01/ybmiao/output/Exhaust/cutler/annotations/instre_m_fixsize480_tau0.15_N3_0_50.json"
    # annotation_file = "/media/media01/ybmiao/output/Exhaust/cutler/annotations/instre_s1_fixsize480_tau0.15_N3_0_100.json"
    # annotation_file = "/media/media01/ybmiao/output/Exhaust/cutler/annotations/instre_s2_fixsize480_tau0.15_N3_0_100.json"
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    # print(imgIds)
    # print(len(imgIds))
    # exit()
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        img_path = os.path.join("/home/ybmiao/data/INSTRE/INSTRE-M",img['file_name'])
        # print(img['file_name'])
        # print(imgId)
        # print(img_path)
        if not os.path.exists(img_path):
            raise ValueError("no img")
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        prm_list = []
        box_list = []
        if len(annIds) > 0:
            for i in range(len(anns)):
                mask = coco.annToMask(anns[i])

                plt.imshow(Image.open(img_path))
                plt.show()
                plt.imshow(mask)
                plt.show()
                exit()


                prm_list.append(mask)
                box_list.append(get_largest_cc_box(mask))
        save_prm_and_box_list(dataset=dataset,prm_list=prm_list,box_list=box_list,fsave=dst_f,mask_save_path=os.path.join(dst_root,"mask"), img_path=img_path)
elif dataset == "Ins160":
    dst_root = "/home/ybmiao/output/Exhaust/cutler/Ins160"
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    dst_f = open(os.path.join(dst_root,"box_ref.txt"),"a")
    annotation_file = "/media/media01/ybmiao/output/Exhaust/cutler/annotations/Ins160_fixsize480_tau0.15_N3_0_160.json"
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        img_path = os.path.join("/home/ybmiao/data/Instance-160/Images",img['file_name'])
        # print(img['file_name'])
        # print(imgId)
        # print(img_path)
        # exit()
        if not os.path.exists(img_path):
            raise ValueError("no img")
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        prm_list = []
        box_list = []
        if len(annIds) > 0:
            for i in range(len(anns)):
                mask = coco.annToMask(anns[i])
                prm_list.append(mask)
                box_list.append(get_largest_cc_box(mask))
        save_prm_and_box_list(dataset=dataset,prm_list=prm_list,box_list=box_list,fsave=dst_f,mask_save_path=os.path.join(dst_root,"mask"), img_path=img_path)
elif dataset == "Ins335":
    dst_root = "/home/ybmiao/output/Exhaust/cutler/Ins335"
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    dst_f = open(os.path.join(dst_root,"box_ref.txt"),"a")
    annotation_file = "/home/ybmiao/output/Exhaust/cutler/annotations/imagenet_train_fixsize480_tau0.15_N3_0_335.json"
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        img_path = os.path.join("/home/ybmiao/data/Instance-335/Images",img['file_name'])
        # print(img['file_name'])
        # print(imgId)
        # print(img_path)
        # exit()
        if not os.path.exists(img_path):
            raise ValueError("no img")
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        prm_list = []
        box_list = []
        if len(annIds) > 0:
            for i in range(len(anns)):
                mask = coco.annToMask(anns[i])
                prm_list.append(mask)
                box_list.append(get_largest_cc_box(mask))
        save_prm_and_box_list(dataset=dataset,prm_list=prm_list,box_list=box_list,fsave=dst_f,mask_save_path=os.path.join(dst_root,"mask"), img_path=img_path)
else:
    raise NotImplementedError
