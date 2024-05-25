import os
import yaml
import cv2
import pickle
import torch
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
from yb_utils import get_bboxes_from_file, calculate_resized_w_h, load_config_file, load_mask, load_roi_mask,\
    downsample_mask, save_bi_np_matrix, load_bi_np_matrix, load_str_np_matrix, save_str_list, load_grd_file,\
    load_box, append_xlsx, create_xlsx
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn import preprocessing
import argparse


class Calculate_IOU():
    def __init__(self, config_path="/home/ybmiao/code/SIS/feature/configs_new.yaml", use_config=True, show_info=True,
                 gpu='1', dataset=None, box_path=None, save_path=None, xlsx_path=None, para='pretrain',
                 mode='all', resized=True, layers=None, pooling=None, stride=32, net='resnet50', methods=None, gt_result=False):
        if use_config:
            config = load_config_file(config_path)
            self.gpu = config['gpu']
            self.dataset = config['dataset']
            self.box_path = config['box_path']
            self.save_path = config['save_path']
            self.para = config['para']
            self.mode = config['mode']
            self.resized = config['resized']
            self.layers = config['layers']
            self.pooling = config['pooling']
            self.stride = config['stride']
            self.net = config['net']
            self.methods = config['methods']
            self.xlsx_path = config['xlsx_path']
        else:
            self.gpu = gpu
            self.dataset = dataset
            self.box_path = box_path
            self.save_path = save_path
            self.para = para
            self.mode = mode
            self.resized = resized
            self.layers = layers
            self.pooling = pooling
            self.stride = stride
            self.net = net
            self.methods = methods
            self.xlsx_path = xlsx_path
        self.device = torch.device("cuda:"+self.gpu)
        self.features = {}
        self.box_info = []
        self.show_info = show_info
        self.Ins160_img_qry = "/home/ybmiao/yb_data/Ins160-img-qry.txt"
        self.Ins335_img_qry = "/home/ybmiao/yb_data/Ins335-img-qry.txt"
        self.instre_img_qry = "/home/ybmiao/yb_data/instre-img-qry.txt"
        self.gt_result = gt_result
        self.Ins160_img_ref = "/media/media01/qysun/data/Instance-160/ref_box_list.txt"
        self.Ins335_img_ref = "/media/media01/qysun/data/Instance-335/ref_box_list.txt"
        self.instre_img_ref = "/media/media01/qysun/data/INSTRE/ref_box_list.txt"

        # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
    
    def calculate_iou(self):
        pass




parse = argparse.ArgumentParser(description='Extract features')

parse.add_argument('--use_config', default=False, action='store_true')
parse.add_argument('--show_info', default=False, action='store_true')
parse.add_argument('--gpu', type=str, default='0')
# parse.add_argument('--dataset', type=str, default='Ins160')
# parse.add_argument('--box_path', type=str, default='/media/media01/ybmiao/output/eig_back/6_29/Ins160/box_ref.txt')
# parse.add_argument('--save_path', type=str, default='/media/media01/ybmiao/output/eig_back/6_29/Ins160/test')
parse.add_argument('--dataset', type=str, default='instre')
parse.add_argument('--box_path', type=str, default='/media/media01/ybmiao/output/eig_back/6_29/instre/box_ref.txt')
parse.add_argument('--save_path', type=str, default='/media/media01/ybmiao/output/eig_back/6_29/instre/test')
parse.add_argument('--para', type=str, default='pretrained')  # swav
parse.add_argument('--mode', type=str, default='all')  # qry / ref / all
parse.add_argument('--resized', default=False, action='store_true')
parse.add_argument('--layers', type=str, nargs='+', default=['layer4.0'])
parse.add_argument('--pooling', type=str, nargs='+', default=['mean','max'])  # 'max')
parse.add_argument('--stride', type=int, default=32)  # 16
parse.add_argument('--net', type=str, default='resnet50')
parse.add_argument('--methods', type=str, nargs='+', default=['roi','mask_roi','expand_roi'])
parse.add_argument('--function', type=str, default='gen_pca_pkl')
parse.add_argument('--xlsx_path', type=str, default='/media/media01/ybmiao/output/eig_back/6_29/')
parse.add_argument('--gt_result', default=False, action='store_true')


if __name__ == "__main__":
    args = parse.parse_args()
    print(args)

    args.use_config = False
    cal_iou = Calculate_IOU(use_config=args.use_config, show_info=args.show_info, gpu=args.gpu, dataset=args.dataset, box_path=args.box_path,
                                         save_path=args.save_path, xlsx_path=args.xlsx_path, para=args.para, mode=args.mode, resized=args.resized,layers=args.layers,
                                         pooling=args.pooling, stride=args.stride, net=args.net, methods=args.methods, gt_result=args.gt_result)
    
    cal_iou.calculate_iou()

# ref_path = "/media/media01/ybmiao/output/dasr/6_6/Ins160/DASR/box-ref.txt"
    # ref_path = "/media/media01/ybmiao/output/dasr/6_6/Ins335/DASR/box-ref.txt"
    # ref_path = "/media/media01/ybmiao/output/dasr/6_6/instre/DASR/box-ref.txt"

    # ref_path = "/media/media01/ybmiao/output/dasr/6_6/Ins160/dasr2/box-ref.txt"
    # ref_path = "/media/media01/ybmiao/output/dasr/6_6/Ins335/dasr2/box-ref.txt"
    # ref_path = "/media/media01/ybmiao/output/dasr/6_6/instre/dasr2/box-ref.txt"