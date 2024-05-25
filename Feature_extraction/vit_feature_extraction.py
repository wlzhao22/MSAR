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
from yb_utils import get_bboxes_from_file, calculate_resized_w_h, load_config_file, load_mask, load_roi_mask,\
    downsample_mask, save_bi_np_matrix, load_bi_np_matrix, load_str_np_matrix, save_str_list, save_str_line


class Feature_Extraction():
    def __init__(self, config_path="/home/ybmiao/code/EMB/feature_extraction/configs.yaml", use_config=True, show_info=True,
                 gpu='1', dataset=None, box_path=None, save_path=None, para='pretrain',
                 mode='all', resized=True, layers=None, pooling=None, stride=32, net='resnet50', methods=None, gt_result=False):
        if use_config:
            config = load_config_file(config_path)
            print(config)
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
        self.features = {}
        self.box_info = []
        self.show_info = show_info
        self.model = self.load_model()
        self.Ins160_img_qry = "/home/ybmiao/yb_data/Ins160-img-qry.txt"
        self.Ins335_img_qry = "/home/ybmiao/yb_data/Ins335-img-qry.txt"
        self.instre_img_qry = "/home/ybmiao/yb_data/instre-img-qry.txt"
        self.gt_result = gt_result
        self.Ins160_img_ref = "/media/media01/qysun/data/Instance-160/ref_box_list.txt"
        self.Ins335_img_ref = "/media/media01/qysun/data/Instance-335/ref_box_list.txt"
        self.instre_img_ref = "/media/media01/qysun/data/INSTRE/ref_box_list.txt"

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

    def load_model(self):
        # 加载模型
        if self.net == "resnet50":
            if self.para == "pretrained":
                print("Loading Pretrain Resnet50......")
                model = models.resnet50(pretrained=True)
            elif self.para == "swav":
                print("Loading Swav Resnet50......")
                model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            else:
                raise ValueError("Input unknown net parameter!")
        elif self.net == "resnet101":
            model = models.resnet101(pretrained=True)
        elif self.net == "dino":
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        else:
            model = None
            raise ValueError("Input unknown net architecture!")
        model.eval().cuda()

        def hook(layer_name):
            def fn(module, input, output):
                self.features[layer_name] = output
            return fn

        # 对目标层使用hook
        handles = []
        for name, module in model.named_modules():
            if name in self.layers:
                handle = module.register_forward_hook(hook(name))
                handles.append(handle)
        return model


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

    def load_mmaskes(self, pic_name,raw_img,boxes, is_qry=None):
        assert is_qry is not None

        # get label
        if self.dataset == "Ins160" or self.dataset == "Ins335":
            label = pic_name
        elif self.dataset == "instre":
            label = pic_name.split('.')[0].split('/')[-3:]
            label = "/".join(label)
        else:
            raise ValueError("Unknown dataset!")

        # 给出一个
        if is_qry:
            qry_mask_output = "/home/ybmiao/output/qry_mask"
            qry_mask_output = os.path.join(qry_mask_output, self.dataset)
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
            path = "/".join(self.box_path.split('/')[:-1]) + "/mask/" + pic_name.replace("/", "_") + ".npy"
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

    def get_info_and_img(self, item):
        if self.dataset == "Ins160" or self.dataset == "Ins335":
            # 获得文件名、用PIL读取图片、resize、ToTensor
            pic_name = str(item).split('.')[0].split('/')[-1]
            file_name = str(item)
            raw_img = PIL.Image.open(file_name).convert('RGB')
        elif self.dataset == "instre":
            pic_name = '/'.join(str(item).split('/')[-3:]).split('.')[0]
            file_name = str(item)
            raw_img = PIL.Image.open(file_name).convert('RGB')
        else:
            raise ValueError("Not dataset dealing in this Function!")
        return pic_name, file_name, raw_img

    def get_box_info(self, boxes, pic_name, is_qry=None):
        assert is_qry is not None
        box_lines = []
        for box_idx, box in enumerate(boxes):
            ori_box = boxes[box_idx]
            if self.dataset == 'Ins160':
                pic_name = pic_name.split('.')[0].split('/')[-1]

            if is_qry is False:
                box_line = pic_name + '>%04d ' % (box_idx + 1) + ' '.join(ori_box.astype(str)) + '\n'
                box_lines.append(box_line)
            elif is_qry is True:
                box_line = pic_name + ' ' + ' '.join(np.asarray(ori_box).astype(str)) + '\n'
                box_lines.append(box_line)
        return box_lines

    def method_resize(self, layer, boxes, rate, pooling, pic_name, raw_img, is_qry=None):
        assert is_qry is not None
        feature_map = self.features[layer]
        if '4' in layer:
            self.stride = 32
        elif '3' in layer:
            self.stride = 16
        else:
            raise ValueError('Unknown layers input!')
        features = []
        box_lines = []
        for box_idx, box in enumerate(boxes):
            # resize box
            ori_box = boxes[box_idx]
            box = np.array([np.floor(ori_box[0] / rate / float(self.stride)),
                            np.floor(ori_box[1] / rate / float(self.stride)),
                            np.ceil(ori_box[2] / rate / float(self.stride)),
                            np.ceil(ori_box[3] / rate / float(self.stride))])
            box = map(int, box)
            xmin, ymin, xmax, ymax = box
            if xmin == xmax or ymin == ymax:
                continue

            if xmin > feature_map.shape[3] or ymin > feature_map.shape[2]:
                print(ori_box)
                print(rate)
                print(xmin, ymin, xmax, ymax)
                print(feature_map.shape)
                print(f'Wrong annotation: {pic_name}>{box_idx}')
                continue
            xmin = max(0, xmin)
            ymin = max(0, ymin)

            if pooling == 'mean':
                feature = torch.mean(feature_map[:, :, ymin:ymax, xmin:xmax], dim=[2, 3]).reshape(-1).cpu().numpy()
            elif pooling == 'max':
                feature, _ = torch.max(feature_map[:, :, ymin:ymax, xmin:xmax], dim=-2, keepdim=True)
                feature, _ = torch.max(feature, dim=-1)
                feature = feature.reshape(-1).cpu().numpy()
            elif pooling == "gem":
                fea_pow = torch.pow(feature_map[:, :, ymin:ymax, xmin:xmax], 3)
                fea_mean = torch.mean(fea_pow, dim=[2, 3]).reshape(-1)
                feature = torch.pow(fea_mean, 1/3).cpu().numpy()
            else:
                raise ValueError('Wrong pooling type!')

            features.append(feature)

            if self.dataset == 'Ins160':
                pic_name = pic_name.split('.')[0].split('/')[-1]

            if is_qry is False:
                box_line = pic_name + '>%04d ' % (box_idx + 1) + ' '.join(ori_box.astype(str)) + '\n'
                box_lines.append(box_line)
            elif is_qry is True:
                box_line = pic_name + ' ' + ' '.join(np.asarray(ori_box).astype(str)) + '\n'
                box_lines.append(box_line)

        return np.array(features), box_lines

    def method_roi(self, layer, boxes, rate, pooling, pic_name, raw_img, is_qry=None):
        assert is_qry is not None
        feature_map = self.features[layer]
        if '4' in layer:
            self.stride = 32
        elif '3' in layer:
            self.stride = 16
        else:
            raise ValueError('Unknown layers input!')
        features = []
        box_lines = []
        for box_idx, box in enumerate(boxes):
            # resize box
            ori_box = boxes[box_idx]
            box = np.array([np.floor(ori_box[0] / rate / float(self.stride)),
                            np.floor(ori_box[1] / rate / float(self.stride)),
                            np.ceil(ori_box[2] / rate / float(self.stride)),
                            np.ceil(ori_box[3] / rate / float(self.stride))])
            box = map(int, box)
            xmin, ymin, xmax, ymax = box
            if xmin == xmax or ymin == ymax:
                continue

            if xmin > feature_map.shape[3] or ymin > feature_map.shape[2]:
                print(ori_box)
                print(rate)
                print(xmin, ymin, xmax, ymax)
                print(feature_map.shape)
                print(f'Wrong annotation: {pic_name}>{box_idx}')
                continue
            xmin = max(0, xmin)
            ymin = max(0, ymin)

            if pooling == 'mean':
                feature = torch.mean(feature_map[:, :, ymin:ymax, xmin:xmax], dim=[2, 3]).reshape(-1).cpu().numpy()
            elif pooling == 'max':
                feature, _ = torch.max(feature_map[:, :, ymin:ymax, xmin:xmax], dim=-2, keepdim=True)
                feature, _ = torch.max(feature, dim=-1)
                feature = feature.reshape(-1).cpu().numpy()
            elif pooling == "gem":
                fea_pow = torch.pow(feature_map[:, :, ymin:ymax, xmin:xmax], 3)
                fea_mean = torch.mean(fea_pow, dim=[2, 3]).reshape(-1)
                feature = torch.pow(fea_mean, 1/3).cpu().numpy()
            else:
                raise ValueError('Wrong pooling type!')

            features.append(feature)

            if self.dataset == 'Ins160':
                pic_name = pic_name.split('.')[0].split('/')[-1]

            if is_qry is False:
                box_line = pic_name + '>%04d ' % (box_idx + 1) + ' '.join(ori_box.astype(str)) + '\n'
                box_lines.append(box_line)
            elif is_qry is True:
                box_line = pic_name + ' ' + ' '.join(np.asarray(ori_box).astype(str)) + '\n'
                box_lines.append(box_line)

        return np.array(features), box_lines

    def method_maskroi(self, layer, boxes, rate, pooling, pic_name, raw_img, is_qry=None):
        assert is_qry is not None
        feature_map = self.features[layer]
        if '4' in layer:
            self.stride = 32
        elif '3' in layer:
            self.stride = 16
        else:
            raise ValueError('Unknown layers input!')

        maskes = self.load_mmaskes(pic_name, raw_img, boxes, is_qry)

        features = []
        box_lines = []
        for box_idx, box in enumerate(boxes):
            mmask = maskes[box_idx]
            ori_box = boxes[box_idx]

            show_mask = False
            if show_mask:
                def show_mask(mask, ax, random_color=False):
                    if random_color:
                        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                    else:
                        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
                    h, w = mask.shape[-2:]
                    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                    ax.imshow(mask_image)

                plt.imshow(raw_img)
                show_mask(mmask, plt.gca())
                plt.show()

            mmask = downsample_mask(mmask, h=feature_map.shape[-2], w=feature_map.shape[-1])

            cites = np.nonzero(mmask)
            if np.size(cites[0]) == 0:
                continue

            fea_list = []
            for i in range(np.size(cites[0])):
                fea_list.append(feature_map[0, :, cites[0][i], cites[1][i]])
            fea = torch.stack(fea_list, dim=0)

            if pooling == 'mean':
                feature = torch.mean(fea, dim=0).reshape(-1).cpu().numpy()
            elif pooling == 'max':
                # feature = torch.max(fea, dim=0).reshape(-1).cpu().numpy()
                feature = torch.max(fea, dim=0, keepdim=True).values.reshape(-1).cpu().numpy()
            elif pooling == "gem":
                fea_pow = torch.pow(fea, 3)
                fea_mean = torch.mean(fea_pow, dim=0).reshape(-1)
                feature = torch.pow(fea_mean, 1/3).cpu().numpy()
            else:
                raise ValueError('Wrong pooling type!')

            features.append(feature)

            if self.dataset == 'Ins160':
                pic_name = pic_name.split('.')[0].split('/')[-1]

            if is_qry is False:
                box_line = pic_name + '>%04d ' % (box_idx + 1) + ' '.join(ori_box.astype(str)) + '\n'
                box_lines.append(box_line)
            elif is_qry is True:
                box_line = pic_name + ' ' + ' '.join(np.asarray(ori_box).astype(str)) + '\n'
                box_lines.append(box_line)

        return np.array(features), box_lines


    def method_expand_roi(self, layer, boxes, rate, pooling, pic_name, raw_img, is_qry=None):
        def expend_roi(xmin, ymin, xmax, ymax, feature_map, expend_scpoe=1):
            # # padding 加3， 比3小则使用新边界
            if xmin > expend_scpoe:
                xmin -= expend_scpoe
            else:
                xmin = 0
            if ymin > expend_scpoe:
                ymin -= expend_scpoe
            else:
                ymin = 0
            if xmax < feature_map.shape[3] - expend_scpoe + 1:
                xmax += expend_scpoe
            else:
                xmax = feature_map.shape[3]
            if ymax < feature_map.shape[2] - expend_scpoe + 1:
                ymax += expend_scpoe
            else:
                ymax = feature_map.shape[2]
            return xmin, ymin, xmax, ymax


        feature_map = self.features[layer]
        if '4' in layer:
            self.stride = 32
        elif '3' in layer:
            self.stride = 16
        else:
            raise ValueError('Unknown layers input!')
        features = []
        box_lines = []
        for box_idx, box in enumerate(boxes):
            # resize box
            ori_box = boxes[box_idx]
            box = np.array([np.floor(ori_box[0] / rate / float(self.stride)),
                            np.floor(ori_box[1] / rate / float(self.stride)),
                            np.ceil(ori_box[2] / rate / float(self.stride)),
                            np.ceil(ori_box[3] / rate / float(self.stride))])
            box = map(int, box)
            xmin, ymin, xmax, ymax = box
            xmin, ymin, xmax, ymax = expend_roi(xmin, ymin, xmax, ymax, feature_map, expend_scpoe=1)

            if xmin == xmax or ymin == ymax:
                continue

            if xmin > feature_map.shape[3] or ymin > feature_map.shape[2]:
                print(ori_box)
                print(rate)
                print(xmin, ymin, xmax, ymax)
                print(feature_map.shape)
                print(f'Wrong annotation: {pic_name}>{box_idx}')
                continue
            xmin = max(0, xmin)
            ymin = max(0, ymin)

            if pooling == 'mean':
                feature = torch.mean(feature_map[:, :, ymin:ymax, xmin:xmax], dim=[2, 3]).reshape(-1).cpu().numpy()
            elif pooling == 'max':
                feature, _ = torch.max(feature_map[:, :, ymin:ymax, xmin:xmax], dim=-2, keepdim=True)
                feature, _ = torch.max(feature, dim=-1)
                feature = feature.reshape(-1).cpu().numpy()
            elif pooling == "gem":
                fea_pow = torch.pow(feature_map[:, :, ymin:ymax, xmin:xmax], 3)
                fea_mean = torch.mean(fea_pow, dim=[2, 3]).reshape(-1)
                feature = torch.pow(fea_mean, 1/3).cpu().numpy()
            else:
                raise ValueError('Wrong pooling type!')

            features.append(feature)

            if self.dataset == 'Ins160':
                pic_name = pic_name.split('.')[0].split('/')[-1]

            if is_qry is False:
                box_line = pic_name + '>%04d ' % (box_idx + 1) + ' '.join(ori_box.astype(str)) + '\n'
                box_lines.append(box_line)
            elif is_qry is True:
                box_line = pic_name + ' ' + ' '.join(np.asarray(ori_box).astype(str)) + '\n'
                box_lines.append(box_line)

        return np.array(features), box_lines


    def extract_feature(self, file_names, file_boxes, is_qry=None):
        if is_qry:
            print('Start extracting qry features ...')
        elif not is_qry:
            print('Start extracting ref features ...')

        box_lines = []
        if self.show_info:
            enum = tqdm(file_names)
        else:
            enum = file_names


        # 为了加快文件载入的速度，提前打开这些文件，并存到字典里
        f_dict = {}
        f_box_dict = {}
        for method in self.methods:
            for layer in self.layers:
                for pooling in self.pooling:
                    name = method + layer + pooling
                    folder_name = self.gen_folder_name(layer=layer, method=method, pooling=pooling)
                    folder_path = os.path.join(self.save_path, folder_name)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    if is_qry:
                        ftr_save_path = os.path.join(folder_path, 'qry-ftr.txt')
                        box_save_path = os.path.join(folder_path, 'qry-box.txt')
                    else:
                        ftr_save_path = os.path.join(folder_path, 'ref-ftr.txt')
                        box_save_path = os.path.join(folder_path, 'ref-box.txt')
                    f_dict[name] = open(ftr_save_path, 'wb')
                    f_box_dict[name] = open(box_save_path, 'w')


        # 遍历每一张图片
        for idx, item in enumerate(enum):
            pic_name, file_name, raw_img = self.get_info_and_img(item)
            # 图片裁剪
            if self.resized:
                w_resized, h_resized, rate = calculate_resized_w_h(raw_img.size[0], raw_img.size[1])
                img_trans = T.Compose([
                    T.Resize([h_resized, w_resized]),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                w_resized, h_resized, rate = raw_img.size[0], raw_img.size[1], 1
                img_trans = T.Compose([
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            input_var = img_trans(raw_img).unsqueeze(0).cuda()

            # 正向传播
            output = self.model(input_var)

            boxes = np.array(file_boxes[idx])
            # # 获得box信息，存到类的列表中
            # box_info = self.get_box_info(boxes, pic_name, is_qry)
            # for b in box_info:
            #     self.box_info.append(b)

            # 遍历不同方法、层、池化方法
            for method in self.methods:
                for layer in self.layers:
                    for pooling in self.pooling:
                        name = method + layer + pooling

                        if method == 'roi':
                            features, boxes_list = self.method_roi(layer,boxes,rate,pooling, pic_name,raw_img, is_qry)
                        elif method == 'mask_roi':
                            features, boxes_list = self.method_maskroi(layer,boxes,rate,pooling, pic_name,raw_img, is_qry)
                        elif method == 'expand_roi':
                            features, boxes_list = self.method_expand_roi(layer,boxes,rate,pooling, pic_name,raw_img, is_qry)
                        # elif method == 'mix':
                        #     pass
                        else:
                            raise ValueError("Unknown method in Function: extract_feature!")
                        # 写入当文件里，传入的是字典的value，对应的是已打开的文件
                        save_bi_np_matrix(features,f_dict[name])
                        save_str_list(boxes_list, f_box_dict[name])

            # 清空存储多层feature_map的字典
            self.features = {}

        # 存入box信息，并且逐个关闭字典中的存feature的文件
        for method in self.methods:
            for layer in self.layers:
                for pooling in self.pooling:
                    name = method + layer + pooling
                    folder_name = self.gen_folder_name(layer=layer, method=method, pooling=pooling)
                    folder_path = os.path.join(self.save_path, folder_name)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    if is_qry:
                        ftr_save_path = os.path.join(folder_path, 'qry-ftr.txt')
                        box_save_path = os.path.join(folder_path, 'qry-box.txt')
                    else:
                        ftr_save_path = os.path.join(folder_path, 'ref-ftr.txt')
                        box_save_path = os.path.join(folder_path, 'ref-box.txt')
                    f_dict[name].close()
                    f_box_dict[name].close()
                    # save_str_list(self.box_info, box_save_path)

    def extract_qry(self):
        '''
        处理Qry文件并提取Qry的特征
        :return:
        '''
        # 获得文件和文本框
        if self.dataset == "Ins160":
            file_name = self.Ins160_img_qry
        elif self.dataset == "Ins335":
            file_name = self.Ins335_img_qry
        elif self.dataset == "instre":
            file_name = self.instre_img_qry
        else:
            raise ValueError("Wrong dataset choose in config.yaml. Please input Ins160/ Ins335/ instre !")

        names, file_boxes = get_bboxes_from_file(file_name)
        file_names = []
        for name in names:
            pic_name = str(name).split('_')[0]
            if self.dataset == "Ins160":
                nn = '/home/ybmiao/data/Instance-160/Images/' + pic_name + '/' + str(name) + '.jpg'
            elif self.dataset == "Ins335":
                nn = '/home/ybmiao/data/Instance-335/Images/' + pic_name + '/' + str(name) + '.jpg'
            elif self.dataset == "instre":
                nn = '/home/ybmiao/data/INSTRE/' + str(name) + '.jpg'
            else:
                raise ValueError("Wrong dataset choose in config.yaml. Please input Ins160/ Ins335/ instre !")
            file_names.append(nn)

        # 提取特征
        with torch.no_grad():
            self.extract_feature(file_names,file_boxes,is_qry=True)


    def extract_ref(self):
        '''
        处理Ref文件并提取Ref的特征
        :return:
        '''
        # 获得文件和文本框
        if self.gt_result:
            if self.dataset == "Ins160":
                file_name = self.Ins160_img_ref
            elif self.dataset == "Ins335":
                file_name = self.Ins335_img_ref
            elif self.dataset == "instre":
                file_name = self.instre_img_ref
            else:
                raise NotImplementedError
            file_names, file_boxes = get_bboxes_from_file(file_name)
        else:
            file_name = self.box_path
            file_names, file_boxes = get_bboxes_from_file(file_name)

        # print(len(file_names))
        # print(file_names[0])
        # print(len(file_boxes))
        # print(file_boxes[0])

        # 提取特征
        with torch.no_grad():
            self.extract_feature(file_names,file_boxes,is_qry=False)

    def extract(self):
        '''
        根据不同的模式提取特征
        :return:
        '''
        if self.mode == "qry":
            self.extract_qry()
        elif self.mode == "ref":
            self.extract_ref()
        elif self.mode == "all":
            self.extract_qry()
            self.extract_ref()
        else:
            raise ValueError("Wrong mode choose in config.yaml. Please input qry/ ref/ all!")




parse = argparse.ArgumentParser(description='Extract features')

parse.add_argument('--use_config', default=False, action='store_true')
parse.add_argument('--show_info', default=False, action='store_true')
parse.add_argument('--gpu', type=str, default='1')
parse.add_argument('--dataset', type=str, default='Ins160')
parse.add_argument('--box_path', type=str, default='/media/media01/ybmiao/output/eig_back/6_29/Ins160/box_ref.txt')
parse.add_argument('--save_path', type=str, default='/media/media01/ybmiao/output/eig_back/6_29/Ins160/test')
parse.add_argument('--para', type=str, default='pretrained')  # swav
parse.add_argument('--mode', type=str, default='all')  # qry / ref / all
parse.add_argument('--resized', default=False, action='store_true')
parse.add_argument('--layers', type=str, nargs='+', default=['layer4.0'])
parse.add_argument('--pooling', type=str, nargs='+', default=['mean','max'])  # 'max')
parse.add_argument('--stride', type=int, default=32)  # 16
parse.add_argument('--net', type=str, default='dino')
# parse.add_argument('--methods', type=str, nargs='+', default=['roi','mask_roi','expand_roi'])
parse.add_argument('--methods', type=str, nargs='+', default=['roi','expand_roi'])
parse.add_argument('--gt_result', default=False, action='store_true')

if __name__ == "__main__":
    args = parse.parse_args()
    print(args)

    # args.use_config = True
    feature_extract = Feature_Extraction(use_config=args.use_config, show_info=args.show_info, gpu=args.gpu, dataset=args.dataset, box_path=args.box_path,
                                         save_path=args.save_path, para=args.para, mode=args.mode, resized=args.resized,layers=args.layers,
                                         pooling=args.pooling, stride=args.stride, net=args.net, methods=args.methods, gt_result=args.gt_result)
    feature_extract.extract()
