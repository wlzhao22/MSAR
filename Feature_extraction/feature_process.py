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


class Feature_Process():
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

    def gen_folder_name(self, layer, method, pooling):
        '''
        根据对应的不同的 layer 和 method 生成对应的路径名称
        :param layer:
        :param method:
        :return:
        '''
        if self.resized:
            resize = "resized"
        else:
            resize = "noresize"
        folder_name = "_".join(
            [self.dataset, self.para, method, resize, layer.replace('.', '-'), pooling])
        ssave_path = self.save_path + "/" + folder_name
        return ssave_path

    def feature_normalization(self, features):
        return preprocessing.normalize(features, norm='l2', axis=1)

    def gen_pca_pkl(self):
        for method in self.methods:
            for layer in self.layers:
                for pooling in self.pooling:
                    name = method + layer + pooling
                    print("Loading Feature Files...")
                    folder_name = self.gen_folder_name(layer,method,pooling)
                    print("Source Path:",folder_name)

                    print("Loading Query Matrix...")
                    qrf_src_file = os.path.join(folder_name, 'qry-ftr.txt')
                    qry_feature = load_bi_np_matrix(qrf_src_file)

                    print("Loading Reference Matrix...")
                    ref_src_file = os.path.join(folder_name, 'ref-ftr.txt')
                    ref_feature = load_bi_np_matrix(ref_src_file)


                    dim = ref_feature.shape[1]
                    pickle_file = os.path.join(folder_name,
                                               "pca-" + folder_name.split('/')[-1] + "-" + str(dim) + ".pkl")
                    if not os.path.exists(pickle_file):
                        print('Start PCA ...\n')
                        pca = PCA(n_components=int(dim), whiten=True)  # dimension

                        ref_feature = self.feature_normalization(ref_feature)
                        pca.fit(ref_feature)

                        print("Writing to:",pickle_file)
                        with open(pickle_file, 'wb') as p_f:
                            pickle.dump(pca, p_f)
                    else:
                        print("PCA file is already existed!")


    def pca_whiten(self):
        for method in self.methods:
            for layer in self.layers:
                for pooling in self.pooling:
                    name = method + layer + pooling
                    print("Loading Feature Files...")
                    folder_name = self.gen_folder_name(layer, method, pooling)
                    print("Source Path:", folder_name)

                    print("Loading Query Matrix...")
                    qrf_src_file = os.path.join(folder_name, 'qry-ftr.txt')
                    qrf_dst_file = os.path.join(folder_name, 'qry-ftr-whiten.txt')
                    qry_feature = load_bi_np_matrix(qrf_src_file)

                    print("Loading Reference Matrix...")
                    ref_src_file = os.path.join(folder_name, 'ref-ftr.txt')
                    ref_dst_file = os.path.join(folder_name, 'ref-ftr-whiten.txt')
                    ref_feature = load_bi_np_matrix(ref_src_file)

                    print("Loading PCA File...")
                    dim = ref_feature.shape[1]
                    # 加载交叉数据集PCA文件
                    # folder_name = "/media/media01/ybmiao/output/EMB/7_22/Ins160/test/" + folder_name.split("/")[-1]
                    pickle_file = os.path.join(folder_name,
                                               "pca-" + folder_name.split('/')[-1] + "-" + str(dim) + ".pkl")
                    if self.dataset == "Ins160" or self.dataset == "Ins335" or self.dataset == "CUHK-SYSU":
                        rpl_name = "instre" # 获取交叉数据集PCA文件名
                    elif self.dataset == "instre":
                        rpl_name = "Ins160"
                    else:
                        raise ValueError("Unknown Dataset Name in PCA file (Function:pce_whiten)")

                    pickle_file = pickle_file.replace(self.dataset, rpl_name)
                    with open(pickle_file, 'rb') as f:
                        pca = pickle.load(f)

                    print('Begin pca whitening...')
                    # pca.fit(ref_features)

                    features = np.concatenate([qry_feature, ref_feature])
                    features = self.feature_normalization(features)

                    new_features = pca.transform(features)
                    # norm_features = preprocessing.normalize(new_features, norm='l2', axis=1)
                    norm_features = self.feature_normalization(new_features)

                    print('Finish PCA Whiten...')


                    qry_num = qry_feature.shape[0]

                    with open(qrf_dst_file, 'wb') as f:
                        save_bi_np_matrix(norm_features[:qry_num,:], f)
                    print('Finish writing qry file')

                    with open(ref_dst_file, 'wb') as f:
                        save_bi_np_matrix(norm_features[qry_num:,:], f)
                    print('Finish writing ref file')

    def get_dst(self, folder_path, topk, is_whiten):
        if topk == 0:
            str_topk = 'all'
        else:
            str_topk = str(topk)

        if is_whiten:
            map_dst = folder_path + '/map' + str_topk + '-whiten.txt'
            box_dst = folder_path + '/box' + str_topk + '-whiten.txt'
        else:
            map_dst = folder_path + '/map' + str_topk + '.txt'
            box_dst = folder_path + '/box' + str_topk + '.txt'
        return map_dst, box_dst

    def load_retrieval_data(self, folder_path, is_whiten, dataset):
        gt_path = None
        if dataset == 'Ins160':
            gt_path = '/media/media01/qysun/data/mAP/Ins160-mAP.txt'
        elif dataset == 'Ins335':
            gt_path = '/media/media01/qysun/data/mAP/Ins335-mAP.txt'
        elif dataset == 'instre':
            gt_path = '/media/media01/qysun/data/mAP/instre-mAP.txt'
        elif dataset == 'CUHK-SYSU':
            gt_path = '/media/media01/qysun/data/CUHK-SYSU/INS/SYSU-All-mAP.txt'
        else:
            print("Wrong Dataset Name Input! Please input 'Ins160', 'Ins335' or 'instre'.")
            exit(-1)

        print("Loading ground truth...... from ", gt_path)
        global grd_dict
        grd_dict = load_grd_file(gt_path)

        qimg = folder_path + '/qry-box.txt'
        rimg = folder_path + '/ref-box.txt'
        if is_whiten:
            qftr = folder_path + '/qry-ftr-whiten.txt'
            rftr = folder_path + '/ref-ftr-whiten.txt'
        else:
            qftr = folder_path + '/qry-ftr.txt'
            rftr = folder_path + '/ref-ftr.txt'

        print("Loading qName and qCod...... from ", qimg)
        qName, qCod = load_box(qimg)
        print("Loading rName and rCod...... from ", rimg)
        rName, rCod = load_box(rimg)
        print("Loading qMat...... from ", qftr)
        qMat = load_bi_np_matrix(qftr)
        print("Loading rMat...... from ", rftr)
        rMat = load_bi_np_matrix(rftr)

        # print("Loading qMat...... from ", qftr)
        # rMat = load_bi_np_matrix(qftr)
        # print("Loading rMat...... from ", rftr)
        # qMat = load_bi_np_matrix(rftr)

        print(len(qName))
        print(len(rName))
        print(qMat.shape)
        print(rMat.shape)

        assert (len(qName) == qMat.shape[0])
        assert (len(rName) == rMat.shape[0])
        assert (qMat.shape[1] == rMat.shape[1])

        return qName, qCod, qMat, rName, rCod, rMat, grd_dict

    def retrieval(self, folder_path, is_whiten, topk, threshold, dataset):
        map_dsts = []
        box_dsts = []
        fs = []
        if isinstance(topk, list):
            for t in topk:
                map_dst, box_dst = self.get_dst(folder_path, t, is_whiten)
                map_dsts.append(map_dst)
                box_dsts.append(box_dst)
                f = open(map_dst, 'w')
                fs.append(f)
        else:
            # map_dst, box_dst = get_dst(topk, is_whiten)
            # map_dsts.append(map_dst)
            # box_dsts.append(box_dst)
            # f = open(map_dst, 'a')
            # fs.append(f)
            print("'topk' mast be a lsit!")
            exit(-1)

        qName, qCod, qMat, rName, rCod, rMat, grdDict = self.load_retrieval_data(folder_path, is_whiten, self.dataset)

        # norm_qMat = np.linalg.norm(qMat, axis=1)
        # norm_rMat = np.linalg.norm(rMat, axis=1)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # qMat = torch.from_numpy(qMat).to(device)
        # rMat = torch.from_numpy(rMat).to(device)
        qMat = torch.from_numpy(qMat).to(self.device)
        rMat = torch.from_numpy(rMat).to(self.device)
        qMatShape = qMat.shape
        rMatShape = rMat.shape
        result_matrix = []

        # qry需要逐条传入，不然会爆显存
        for i in range(qMatShape[0]):
            similarity_matrix = torch.nn.functional.cosine_similarity(rMat, qMat[i].unsqueeze(0), dim=1)
            similarity_matrix = similarity_matrix.cpu().numpy()
            result_matrix.append(similarity_matrix)
        result_matrix = np.array(result_matrix)
        print(result_matrix.shape)
        del qMat
        del rMat

        cnt = 0
        rDict = {}
        for i in range(rMatShape[0]):
            ref = rName[i].split('>')[0]
            if rDict.get(ref) is None and ref not in qName:
                rDict.update({ref: cnt})
                cnt += 1
        print("Finding Ref Names:", cnt)

        print("Retrieval......")
        rankDict = {}
        for i in range(qMatShape[0]):
            if i % 100 == 99:
                print("Already retrieval:", i+1)
            simArray = np.zeros(cnt, dtype=np.float32)
            for j in range(rMatShape[0]):
                ref = rName[j].split('>')[0]
                idx = rDict.get(ref)
                if idx is None:
                    continue
                # sim = cosine_similarity(qMat[i].reshape(1, -1), rMat[j].reshape(1, -1))[0][0]
                # sim = qMat[i].dot(rMat[j]) / (norm_qMat[i] * norm_rMat[j])
                sim = result_matrix[i][j]
                if sim > threshold and sim > simArray[idx]:
                # if sim > simArray[idx]:
                    simArray[idx] = sim
                    if rankDict.get(idx) is None:
                        rankDict.update({idx: j})
                    else:
                        rankDict[idx] = j


            index = np.argsort(simArray)[::-1]
            for k in range(len(map_dsts)):
                if topk[k] == 0:
                    x1 = np.where(simArray>0)
                    num = len(x1[0])
                    # num = len(grdDict[qName[i]])
                else:
                    # num = topk[k]
                    num = min(topk[k], len(grdDict[qName[i]]))

                for j in range(num):
                    idx = index[j]
                    # print(qName[i], ' ', rName[rankDict[idx]].split('>')[0], ' ', simArray[j])
                    info = qName[i] + ' ' + rName[rankDict[idx]].split('>')[0] + ' ' + str(simArray[idx]) + '\n'
                    fs[k].writelines(info)

        for f in fs:
            f.close()
        return map_dsts, grdDict, rMatShape[0], rMatShape[1]

    def calculate_map(self, grdDict, map_paths):
        map_list = []
        for map_path in map_paths:
            ap_path = map_path.replace("map", "ap")
            f_ap = open(ap_path, 'w')
            map_dict = {}
            with open(map_path, 'r') as f:
                line = f.readline().strip()
                while line:
                    data = line.split()
                    if map_dict.get(data[0]) is None:
                        map_dict.update({data[0]: np.zeros(3, dtype=np.float32)})
                    arr = map_dict.get(data[0])
                    arr[0] += 1
                    if data[1] in grdDict.get(data[0]):
                        arr[1] += 1
                        arr[2] += arr[1] / arr[0]
                    # print(map_dict.get(data[0]))
                    line = f.readline().strip()
            f.close()
            cnt = 0
            map = 0
            for key_value_pair in map_dict.items():
                cnt += 1
                info1 = "The " + str(cnt) + ' Query: ' + key_value_pair[0]
                print("The " + str(cnt) + ' Query: ' + key_value_pair[0])
                # ap = key_value_pair[1][2] / key_value_pair[1][0]
                if map_path.split('.')[0].split('/')[-1][3] == "a":
                    # print(map_path.split('.')[0].split('/')[-1])
                    # print(map_path.split('.')[0].split('/')[-1])
                    ap = key_value_pair[1][2] / len(grd_dict.get(key_value_pair[0]))
                # elif key_value_pair[1][0]<len(grd_dict.get(key_value_pair[0])):
                #     ap = key_value_pair[1][2] / key_value_pair[1][0]
                else:
                    ap = key_value_pair[1][2] / key_value_pair[1][0]
                # ap = key_value_pair[1][2] / len(grd_dict.get(key_value_pair[0]))
                info2 = "AP:" + str(ap)
                print("AP:", ap)
                f_ap.writelines(info1 + "\n" + info2 + "\n")
                map += ap
            print(map_path.split('.')[0].split('/')[-1] + ': ', map / cnt)
            map_list.append(map / cnt)
        return map_list

    def retrieval_and_map(self,topk=None, is_whiten=True, is_save=True):
        for method in self.methods:
            for layer in self.layers:
                for pooling in self.pooling:
                    name = method + layer + pooling
                    print("Loading Feature Files...")
                    folder_name = self.gen_folder_name(layer, method, pooling)
                    print("Source Path:", folder_name)


                    if is_whiten:
                        qrf_src_file = os.path.join(folder_name, 'qry-ftr-whiten.txt')
                        ref_src_file = os.path.join(folder_name, 'ref-ftr-whiten.txt')
                    else:
                        qrf_src_file = os.path.join(folder_name, 'qry-ftr.txt')
                        ref_src_file = os.path.join(folder_name, 'ref-ftr.txt')

                    print("Loading Query Matrix...")
                    qrf_src_file = os.path.join(folder_name, 'qry-ftr-whiten.txt')
                    qry_feature = load_bi_np_matrix(qrf_src_file)

                    print("Loading Reference Matrix...")
                    ref_src_file = os.path.join(folder_name, 'ref-ftr-whiten.txt')
                    ref_feature = load_bi_np_matrix(ref_src_file)

                    map_dsts, grdDict, num, dim = self.retrieval(folder_name, is_whiten=True, topk=topk, threshold=0,
                                                            dataset=self.dataset)
                    map_list = self.calculate_map(grdDict, map_dsts)
                    if is_save:
                        if not os.path.exists(self.xlsx_path):
                            os.mkdir(self.xlsx_path)
                        xlsx_path = os.path.join(self.xlsx_path, 'result.xlsx')
                        if not os.path.exists(xlsx_path):
                            create_xlsx(xlsx_path)
                        info = [self.para, layer, method, str(self.resized), pooling, str(is_whiten), num, dim]
                        output = info + map_list
                        print("Writing to:", xlsx_path)
                        append_xlsx(xlsx_path, self.dataset, output)

    
    def top1_and_recall1_retrieval_and_map(self,topk=None, is_whiten=True, is_save=True):
        for method in self.methods:
            for layer in self.layers:
                for pooling in self.pooling:
                    name = method + layer + pooling
                    print("Loading Feature Files...")
                    folder_name = self.gen_folder_name(layer, method, pooling)
                    print("Source Path:", folder_name)


                    if is_whiten:
                        qrf_src_file = os.path.join(folder_name, 'qry-ftr-whiten.txt')
                        ref_src_file = os.path.join(folder_name, 'ref-ftr-whiten.txt')
                    else:
                        qrf_src_file = os.path.join(folder_name, 'qry-ftr.txt')
                        ref_src_file = os.path.join(folder_name, 'ref-ftr.txt')

                    print("Loading Query Matrix...")
                    qrf_src_file = os.path.join(folder_name, 'qry-ftr-whiten.txt')
                    qry_feature = load_bi_np_matrix(qrf_src_file)

                    print("Loading Reference Matrix...")
                    ref_src_file = os.path.join(folder_name, 'ref-ftr-whiten.txt')
                    ref_feature = load_bi_np_matrix(ref_src_file)

                    map_dsts, grdDict, num, dim = self.retrieval(folder_name, is_whiten=True, topk=topk, threshold=0,
                                                            dataset=self.dataset)
                    map_list = self.calculate_map(grdDict, map_dsts)
                    if is_save:
                        if not os.path.exists(self.xlsx_path):
                            os.mkdir(self.xlsx_path)
                        xlsx_path = os.path.join(self.xlsx_path, 'result.xlsx')
                        if not os.path.exists(xlsx_path):
                            create_xlsx(xlsx_path)
                        info = [self.para, layer, method, str(self.resized), pooling, str(is_whiten), num, dim]
                        output = info + map_list
                        print("Writing to:", xlsx_path)
                        append_xlsx(xlsx_path, self.dataset, output)



parse = argparse.ArgumentParser(description='Extract features')

parse.add_argument('--use_config', default=False, action='store_true')
parse.add_argument('--show_info', default=False, action='store_true')
parse.add_argument('--gpu', type=str, default='1')
# parse.add_argument('--dataset', type=str, default='Ins160')
# parse.add_argument('--box_path', type=str, default='/media/media01/ybmiao/output/EMB/7_22/Ins160/box_ref.txt')
# parse.add_argument('--save_path', type=str, default='/media/media01/ybmiao/output/EMB/7_22/Ins160/test')
# parse.add_argument('--dataset', type=str, default='instre')
# parse.add_argument('--box_path', type=str, default='/home/ybmiao/output/kcut/10_20/instre/box_ref.txt')
# parse.add_argument('--save_path', type=str, default='/home/ybmiao/output/kcut/10_20/instre/test')
# parse.add_argument('--dataset', type=str, default='Ins160')
# parse.add_argument('--box_path', type=str, default='/home/ybmiao/output/kcut/10_20/Ins160/box_ref.txt')
# parse.add_argument('--save_path', type=str, default='/home/ybmiao/output/kcut/10_20/Ins160/test')
# parse.add_argument('--dataset', type=str, default='instre')
# parse.add_argument('--box_path', type=str, default='/media/media01/ybmiao/output/Exhaust/cutler/instre/box_ref.txt')
# parse.add_argument('--save_path', type=str, default='/media/media01/ybmiao/output/Exhaust/cutler/instre/test')
# parse.add_argument('--dataset', type=str, default='Ins335')
# parse.add_argument('--box_path', type=str, default='/media/media01/ybmiao/output/Exhaust/cutler/Ins335/box_ref.txt')
# parse.add_argument('--save_path', type=str, default='/media/media01/ybmiao/output/Exhaust/cutler/Ins335/test')
# parse.add_argument('--dataset', type=str, default='instre')
# parse.add_argument('--box_path', type=str, default='/home/ybmiao/output/kcut/12_07/instre/box_ref.txt')
# parse.add_argument('--save_path', type=str, default='/home/ybmiao/output/kcut/12_07/instre/test')
# parse.add_argument('--dataset', type=str, default='Ins335')
# parse.add_argument('--box_path', type=str, default='/home/ybmiao/output/kcut/12_07/Ins335/box_ref.txt')
# parse.add_argument('--save_path', type=str, default='/home/ybmiao/output/kcut/12_07/Ins335/test')
parse.add_argument('--dataset', type=str, default='instre')
parse.add_argument('--box_path', type=str, default='/home/ybmiao/output/Exhaust/cutler/instre/box_ref.txt')
parse.add_argument('--save_path', type=str, default='/home/ybmiao/output/Exhaust/cutler/instre/test')
# parse.add_argument('--dataset', type=str, default='Ins160')
# parse.add_argument('--box_path', type=str, default='/home/ybmiao/output/Exhaust/cutler/Ins160/box_ref.txt')
# parse.add_argument('--save_path', type=str, default='/home/ybmiao/output/Exhaust/cutler/Ins160/test')
# parse.add_argument('--dataset', type=str, default='CUHK-SYSU')
# parse.add_argument('--box_path', type=str, default='/media/media01/ybmiao/output/EMB/7_22/CUHK-SYSU/box_ref.txt')
# parse.add_argument('--save_path', type=str, default='/media/media01/ybmiao/output/EMB/7_22/CUHK-SYSU/test')
# parse.add_argument('--dataset', type=str, default='instre')
# parse.add_argument('--box_path', type=str, default='/media/media01/ybmiao/output/eig_back/6_29/instre/box_ref.txt')
# parse.add_argument('--save_path', type=str, default='/media/media01/ybmiao/output/eig_back/6_29/instre/test')
parse.add_argument('--para', type=str, default='pretrained')  # swav
parse.add_argument('--mode', type=str, default='all')  # qry / ref / all
parse.add_argument('--resized', default=False, action='store_true')
# parse.add_argument('--layers', type=str, nargs='+', default=['layer4.0', 'layer4.1'])
parse.add_argument('--layers', type=str, nargs='+', default=['layer4.0'])
# parse.add_argument('--layers', type=str, nargs='+', default=['layer4.1'])
# parse.add_argument('--pooling', type=str, nargs='+', default=['gem', 'mean'])  # 'max')
parse.add_argument('--pooling', type=str, nargs='+', default=['gem'])  # 'max')
parse.add_argument('--stride', type=int, default=32)  # 16
parse.add_argument('--net', type=str, default='resnet50')
# parse.add_argument('--methods', type=str, nargs='+', default=['mask_roi','roi'])
parse.add_argument('--methods', type=str, nargs='+', default=['roi'])
# parse.add_argument('--methods', type=str, nargs='+', default=['mask_roi','roi'])
# parse.add_argument('--function', type=str, default='pca_whiten')
# parse.add_argument('--function', type=str, default='gen_pca_pkl')
parse.add_argument('--function', type=str, default='retrieval_and_map')
# parse.add_argument('--xlsx_path', type=str, default='/home/ybmiao/output/kcut/10_20')
parse.add_argument('--xlsx_path', type=str, default='/home/ybmiao/output/Exhaust/cutler')
parse.add_argument('--gt_result', default=False, action='store_true')


if __name__ == "__main__":
    args = parse.parse_args()
    print(args)

    args.use_config = False
    feature_process = Feature_Process(use_config=args.use_config, show_info=args.show_info, gpu=args.gpu, dataset=args.dataset, box_path=args.box_path,
                                         save_path=args.save_path, xlsx_path=args.xlsx_path, para=args.para, mode=args.mode, resized=args.resized,layers=args.layers,
                                         pooling=args.pooling, stride=args.stride, net=args.net, methods=args.methods, gt_result=args.gt_result)
    if args.function == "gen_pca_pkl":
        feature_process.gen_pca_pkl()
    elif args.function == "pca_whiten":
        feature_process.pca_whiten()
    elif args.function == "retrieval_and_map":
        if args.dataset == 'CUHK-SYSU':
            feature_process.retrieval_and_map(topk=[10], is_save=False)
        else:
            feature_process.retrieval_and_map(topk=[50, 100, 0])
    else:
        raise ValueError("Unknown function input!")



