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

def mask_iou(mask1:np.array, mask2:np.array):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    # 计算 IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou

def cal_recall(num, ann_sum, thre, mask, arr):
    for j in range(num):
        iou_score = mask_iou(mask, arr[j])
        if iou_score > thre:
            # print(iou_score)
            ann_sum += 1
            break
    return ann_sum

def main():
    annotation_file = "/media/media01/qysun/data/coco/annotations/instances_val2017.json"
    img_root = "/home/ybmiao/data/coco/images/val2017"
    # mask_root = "/home/ybmiao/output/Exhaust/dss/coco_val2017/mask"
    # mask_root = "/home/ybmiao/output/Exhaust/cutler/coco_val2017/mask"
    # mask_root = "/home/ybmiao/output/Exhaust/tokencut/coco_val2017/mask"
    file_name_path = "/media/media01/qysun/data/coco/coco_val2017_filename.txt"
    mask_root = "/home/ybmiao/output/kcut/12_07/coco_val2017/mask"


    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    file_names = open(file_name_path,"r").readlines()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    print("file_names len:{}".format(len(file_names)))
    exit()
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

    for idx, imgId in enumerate(tqdm.tqdm(imgIds, ncols=100)):
        img = coco.loadImgs(imgId)[0]
        
        # 先载入对比保存的npy文件
        mask_path = mask_root + "/" + str(img['file_name']).split(".")[0] + ".npy"
        if not os.path.exists(mask_path):
            continue
        arr = np.load(mask_path)
        # print(arr.shape)

        img_path = os.path.join(img_root,img['file_name'])
        f_img = Image.open(img_path,"r")
        # plt.imshow(f_img)
        # plt.show()
        #
        # # for i in range(arr.shape[0]):
        # #     plt.imshow(arr[i])
        # #     plt.show()
        # plt.imshow(arr)
        # plt.show()
        # exit()
        # continue
        if "tokencut" in mask_path:
            imgId = int(file_names[idx].split('.')[0])
            img = coco.loadImgs(imgId)[0]
            arr = arr[np.newaxis, :, :]
        elif "cutler" in mask_path:
            if arr.shape[0] == 0:
                continue
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        gt_mask = []
        if len(annIds) > 0:
            all_ann += len(annIds)
            for i in range(len(anns)):
                # 加载mask

                if "dss" in mask_path:
                    mask = coco.annToMask(anns[i])
                    mask = cv2.resize(mask, (mask.shape[1] // 8, mask.shape[0] // 8), interpolation=cv2.INTER_AREA)

                elif "ksum"  in mask_path:
                    w, h = f_img.size
                    if min(w, h) > 355:
                        transform = pth_transforms.Compose([
                            pth_transforms.Resize(360, max_size=800),
                            pth_transforms.ToTensor(),
                            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
                    else:
                        transform = pth_transforms.Compose([
                            pth_transforms.ToTensor(),
                            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
                    tensor = transform(f_img.convert('RGB'))
                    mask = coco.annToMask(anns[i])
                    mask = cv2.resize(mask, (tensor.shape[2] // 8, tensor.shape[1] // 8), interpolation=cv2.INTER_AREA)

                elif "kcut"  in mask_path:
                    w, h = f_img.size
                    if min(w, h) > 355:
                        transform = pth_transforms.Compose([
                            pth_transforms.Resize(360, max_size=800),
                            pth_transforms.ToTensor(),
                            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
                    else:
                        transform = pth_transforms.Compose([
                            pth_transforms.ToTensor(),
                            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
                    tensor = transform(f_img.convert('RGB'))

                    down_sampled = True
                    # down_sampled = False
                    if down_sampled:
                        mask = coco.annToMask(anns[i])
                        mask = cv2.resize(mask, (tensor.shape[2] // 8, tensor.shape[1] // 8), interpolation=cv2.INTER_AREA)

                        # for i in range(arr.shape[0]):
                        # arr = cv2.resize(arr, (tensor.shape[2] // 8, tensor.shape[1] // 8), interpolation=cv2.INTER_AREA)
                        resized_data = []
                        for i in range(arr.shape[0]):
                            # plt.imshow(arr[i])
                            # plt.show()
                            resized_arr = cv2.resize(arr[i], (tensor.shape[2] // 8, tensor.shape[1] // 8), interpolation=cv2.INTER_AREA)  # 调整到5x5大小
                            # resized_data.append(resized_arr)

                            # plt.imshow(resized_arr)
                            # plt.show()

                            # 定义一个椭圆形的核，可以根据需要调整大小
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                            # 执行闭运算来填充空隙
                            filled_mask = cv2.morphologyEx(resized_arr, cv2.MORPH_CLOSE, kernel)
                            resized_data.append(filled_mask)
                            # plt.imshow(filled_mask)
                            # plt.show()
                            # exit()

                            # 将调整大小后的数据重新组合为3D数组
                        arr = np.stack(resized_data, axis=0)
                    else:
                        mask = coco.annToMask(anns[i])
                        mask = cv2.resize(mask, (tensor.shape[2], tensor.shape[1]),
                                          interpolation=cv2.INTER_AREA)
                        filled_data = []
                        for i in range(arr.shape[0]):
                            # 定义一个椭圆形的核，可以根据需要调整大小
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                            # 执行闭运算来填充空隙
                            filled_mask = cv2.morphologyEx(arr[i], cv2.MORPH_CLOSE, kernel)
                            filled_data.append(filled_mask)
                        arr = np.stack(filled_data, axis=0)


                elif "tokencut" in mask_path:
                    mask = coco.annToMask(anns[i])

                    mask_size = (int(np.ceil(mask.shape[0] / 8) * 8),
                                 int(np.ceil(mask.shape[1] / 8) * 8))
                    paded = np.zeros(mask_size)
                    paded[ :mask.shape[0], : mask.shape[1]] = mask
                    mask = paded
                    mask = cv2.resize(mask, (mask.shape[1] // 8, mask.shape[0] // 8), interpolation=cv2.INTER_AREA)
                    # plt.imshow(mask)
                    # plt.show()

                elif "cutler" in mask_path:
                    mask = coco.annToMask(anns[i])
                else:
                    raise ValueError

                # print(arr.shape)
                # print(mask.shape)
                # 检查mask格式是否对应
                assert arr.shape[1] == mask.shape[0] and arr.shape[2] == mask.shape[1]

                gt_mask.append(mask)

                miou10_ann = cal_recall(arr.shape[0], miou10_ann,0.1,mask,arr)
                miou20_ann = cal_recall(arr.shape[0], miou20_ann, 0.2, mask, arr)
                miou30_ann = cal_recall(arr.shape[0], miou30_ann, 0.3, mask, arr)
                miou40_ann = cal_recall(arr.shape[0], miou40_ann, 0.4, mask, arr)
                miou50_ann = cal_recall(arr.shape[0], miou50_ann, 0.5, mask, arr)
                miou60_ann = cal_recall(arr.shape[0], miou60_ann, 0.6, mask, arr)
                miou70_ann = cal_recall(arr.shape[0], miou70_ann, 0.7, mask, arr)
                miou80_ann = cal_recall(arr.shape[0], miou80_ann, 0.8, mask, arr)
                miou90_ann = cal_recall(arr.shape[0], miou90_ann, 0.9, mask, arr)


        # exit()
    print("miou10:\n",miou10_ann, miou10_ann/all_ann)
    print("miou20:\n",miou20_ann, miou20_ann/all_ann)
    print("miou30:\n",miou30_ann, miou30_ann/all_ann)
    print("miou40:\n",miou40_ann, miou40_ann/all_ann)
    print("miou50:\n",miou50_ann, miou50_ann/all_ann)
    print("miou60:\n",miou60_ann, miou60_ann/all_ann)
    print("miou70:\n",miou70_ann, miou70_ann/all_ann)
    print("miou80:\n",miou80_ann, miou80_ann/all_ann)
    print("miou90:\n",miou90_ann, miou90_ann/all_ann)
    print(all_ann)

if __name__ == '__main__':
    main()
