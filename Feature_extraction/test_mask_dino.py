import os
import yaml
import cv2
import pickle
import torch
import argparse
import numpy
import numpy as np
# import tensorflow as tf
from tqdm import tqdm
from sklearn import preprocessing
from torchvision import models
from torchvision import transforms as T
import matplotlib.pyplot as plt
import torch.utils.data
import PIL.Image
import torchvision.ops as ops
import torch.nn.functional as F
import re
import os
from PIL import Image
import vision_transformer as vits
from yb_utils import get_bboxes_from_file, calculate_resized_w_h, load_config_file, load_mask, load_roi_mask,\
    downsample_mask, save_bi_np_matrix, load_bi_np_matrix, load_str_np_matrix, save_str_list, save_str_line


def load_mmaskes(pic_name, raw_img, boxes, is_qry=None):
    assert is_qry is not None

    # get label

    label = pic_name

    # 给出一个
    if is_qry:
        qry_mask_output = "/home/ybmiao/output/qry_mask"
        qry_mask_output = os.path.join(qry_mask_output, "Ins160")
        path = qry_mask_output + "/{}.txt".format(label.replace("/", "_"))
        mm = load_mask(path)
        ori_box = boxes[0]
        w = ori_box[2] - ori_box[0]
        mask = []
        for i in range(0, len(mm[0]), w):
            row = mm[0][i:i + w]  # 从字符串中取出连续的w个字符作为一行
            mask.append(np.array(list(row), dtype=int))  # 将行添加到矩阵列表中
        # 现在得到的mask是box内的mask，不是原图大小的mask
        mask = np.array(mask)
        maskes = np.zeros((raw_img.size[1], raw_img.size[0]))
        maskes[ori_box[1]: ori_box[3], ori_box[0]: ori_box[2]] = mask
        maskes = np.expand_dims(maskes, axis=0)
    else:
        path = "/".join("/media/media01/ybmiao/output/eig_back/6_29/Ins160/box_ref.txt".split('/')[:-1]) + "/mask/" + pic_name.replace("/", "_") + ".npy"
        maskes = load_roi_mask(path)
        # 将输入矩阵转换为PyTorch张量
        tensor = torch.from_numpy(maskes)
        # 添加批次维度（维度0）
        tensor = tensor.unsqueeze(0)
        # 定义目标上采样大小
        target_height, target_width = raw_img.size[1], raw_img.size[0]
        # 使用插值进行上采样
        upsampled_tensor = F.interpolate(tensor, size=(target_height, target_width), mode='bilinear',
                                         align_corners=False)
        # 移除批次维度
        upsampled_tensor = upsampled_tensor.squeeze(0)
        # 将张量转换回NumPy数组
        maskes = upsampled_tensor.numpy()

    return maskes


file_name = "/media/media01/ybmiao/output/eig_back/6_29/Ins160/box_ref.txt"
file_names, file_boxes = get_bboxes_from_file(file_name)
model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

# model = vits.__dict__['vit_base'](patch_size=8, num_classes=0)
# for p in model.parameters():
#     p.requires_grad = False
# url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
# state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
# model.load_state_dict(state_dict, strict=True)

model.eval()
model.to("cuda:1")

for idx, item in enumerate(tqdm(file_names)):
    pic_name = str(item).split('.')[0].split('/')[-1]
    file_name = str(item)
    raw_img = PIL.Image.open(file_name).convert('RGB')
    w_resized, h_resized, rate = raw_img.size[0], raw_img.size[1], 1
    img_trans = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # w_resized, h_resized, rate = calculate_resized_w_h(raw_img.size[0], raw_img.size[1])
    # img_trans = T.Compose([
    #     T.Resize([h_resized, w_resized]),
    #     T.ToTensor(),
    #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    input_var = img_trans(raw_img).unsqueeze(0).to("cuda:1")
    w, h = input_var.shape[-2] - input_var.shape[-2] % 8, input_var.shape[-1] - input_var.shape[-1] % 8
    input_var = input_var[:, :, :w, :h]
    boxes = np.array(file_boxes[idx])
    maskes = load_mmaskes(pic_name, raw_img, boxes, False)

    print(boxes)
    print(maskes.shape)
    
    for mask in maskes:
        w, h = mask.shape[-2] - mask.shape[-2] % 8, mask.shape[-1] - mask.shape[-1] % 8
        img = mask[:w, :h]


        # 处理mask矩阵
        height, width = mask.shape
        downsampled_height = mask.shape[-2] // 8
        downsampled_width = mask.shape[-1] // 8
        patch_h = 8
        patch_w = 8

        # 创建一个空的下采样掩码矩阵
        downsampled_mask = np.zeros((downsampled_height, downsampled_width), dtype=int)

        # 遍历每个下采样块
        for i in range(downsampled_height):
            for j in range(downsampled_width):
                # 提取当前块的原始掩码块
                block = mask[i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]
                # print(np.sum(block))
                # 检查当前块是否至少存在10个为1的像素
                if np.sum(block) > 10:
                    downsampled_mask[i, j] = 1

        # cites = np.nonzero(downsampled_mask)
        # if np.size(cites[0]) == 0:
        #     raise ValueError
        #
        # h_min = min(cites[0])
        # h_max = max(cites[0])
        # w_min = min(cites[1])
        # w_max = max(cites[1])

        output, attens = model.get_last_output_and_selfattention_with_mask(input_var, downsampled_mask)
        # output, attens = model.get_last_output_and_selfattention(input_var)
        print(output.shape)
        print(attens.shape)
        print(output[0,0,:].shape)
    exit()