import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as pth_transforms
import torch
import os
from accelerate import Accelerator
import sys
import numpy as np
from scipy.linalg import eigh
import torch.nn.functional as F
from tsnecuda import TSNE
from vit_feature import get_dino_output, get_dinov2_output, get_pretrained_output, get_clip_output
from scipy import ndimage
import copy
import time
import math
from yb_utils import show_img_and_box
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import eigh
from scipy import ndimage
import scipy
from VIT_BP import VIT_Backprop


class Graph_Visualization():
    def __init__(self, img_path, model_name, patch_size):
        self.gpu = "1"
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        self.device = torch.device("cuda:" + self.gpu)
        self.model_name = model_name
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        self.model.eval()
        self.model.to(self.device)

        self.bp = VIT_Backprop(self.model)
        self.bp._patch()

        self.ori_img = np.asarray(Image.open(img_path).convert('RGB')).astype(np.uint8)
        self.img, self.rate = self.get_input_var(img_path, show_img=False)
        self.img.requires_grad_()
        self.patch_size = patch_size
        self.h_featmap = self.img.shape[-2] // self.patch_size
        self.w_featmap = self.img.shape[-1] // self.patch_size
        self.dims = [self.h_featmap, self.w_featmap]
        self.indices = np.arange(self.h_featmap * self.w_featmap)

        self.output, attens = self.get_output(self.img, model_name=self.model_name)
        self.output.requires_grad_(True)
        num_heads = self.model.blocks[0].attn.num_heads
        self.attens = attens[0, :, 0, 1:].reshape(num_heads, -1)
        
        self.output = F.normalize(self.output, p=2)
        self.correspend = (self.output @ self.output.transpose(1, 0)).cpu().detach().numpy()
        self.tau = 0.2
        self.eps = 1e-5
        self.no_binary_graph = False
        if self.no_binary_graph:
            self.correspend[self.correspend < self.tau] = self.eps
        else:
            self.correspend = self.correspend > self.tau
            self.correspend = np.where(self.correspend.astype(float) == 0, self.eps, self.correspend)
        self.d_i = np.sum(self.correspend, axis=1)
        self.pseudo_labels = self.init_pseudo_labels(threshold=1, visualize=False)
        self.di_diag = np.diag(self.d_i)

    def update_init_info(self, img_path):
        self.ori_img = np.asarray(Image.open(img_path).convert('RGB')).astype(np.uint8)
        self.img, self.rate = self.get_input_var(img_path, show_img=False)
        self.img.requires_grad_()
        self.patch_size = patch_size
        self.h_featmap = self.img.shape[-2] // self.patch_size
        self.w_featmap = self.img.shape[-1] // self.patch_size
        self.dims = [self.h_featmap, self.w_featmap]
        self.indices = np.arange(self.h_featmap * self.w_featmap)
        
        self.output, attens = self.get_output(self.img, model_name=self.model_name)
        self.output.requires_grad_(True)
        num_heads = self.model.blocks[0].attn.num_heads
        self.attens = attens[0, :, 0, 1:].reshape(num_heads, -1)


        self.output = F.normalize(self.output, p=2)
        self.correspend = (self.output @ self.output.transpose(1, 0)).cpu().detach().numpy()
        self.tau = 0.2
        self.eps = 1e-5
        self.no_binary_graph = False
        if self.no_binary_graph:
            self.correspend[self.correspend < self.tau] = self.eps
        else:
            self.correspend = self.correspend > self.tau
            self.correspend = np.where(self.correspend.astype(float) == 0, self.eps, self.correspend)
        self.d_i = np.sum(self.correspend, axis=1)
        self.pseudo_labels = self.init_pseudo_labels(threshold=1, visualize=False)
        self.di_diag = np.diag(self.d_i)

    def init_pseudo_labels(self, threshold=1, visualize=False):
        pseudo_label = np.zeros(self.d_i.shape)
        thre = sorted(list(self.d_i))[len(self.d_i) * threshold // 10]
        for i in range(self.d_i.shape[0]):
            if self.d_i[i] <= thre:
                pseudo_label[i] = 1
        if visualize:
            mask = pseudo_label.reshape(self.dims)
            plt.imshow(mask)
            plt.show()
        return pseudo_label

    def get_input_var(self, img_path, show_img=False):
        if img_path is not None:
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                if show_img:
                    plt.imshow(img)
                    plt.axis('off')
                    plt.show()
        else:
            print(f"Provided image path {img_path} is non valid.")
            sys.exit(1)

        w, h = img.size
        if min(w, h) > 295:
            transform = pth_transforms.Compose([
                pth_transforms.Resize(300, max_size=600),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            transform = pth_transforms.Compose([
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        img = transform(img)
        rate = h / img.shape[-2]

        # w, h = img.size
        # transform = pth_transforms.Compose([
        #     pth_transforms.Resize((224,224)),
        #     pth_transforms.ToTensor(),
        #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        # img = transform(img)
        # rate = h / img.shape[-2]

        return img, rate

    def get_output(self, img, model_name="dino"):

        img = img.to(self.device)

        # get related parameter
        w, h = img.shape[1] - img.shape[1] % self.patch_size, img.shape[2] - img.shape[2] % self.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        img.requires_grad_()
        attens = None

        if model_name == "dino":
            # output, attens = get_dino_output(img, self.device)
            output, attens = self.model.get_last_output_and_selfattention(img)
            output = output[0, 1:, :]
            output.requires_grad_(True)
            # output = output[0, 1:, :].detach().cpu().numpy()
            # output = output[0, 1:, :].detach()
        elif model_name == "dinov2":
            output = get_dinov2_output(img, self.device)[0][0].detach().cpu().numpy()
        elif model_name == "pretrained":
            output = get_pretrained_output(img, self.device)
        elif model_name == "clip":
            output = get_clip_output(img, self.device)
        else:
            raise NotImplementedError
        return output, attens

    def tsne_visualize(self, category=None):
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'pink', 'brown', 'gray', 'black']

        tsne = TSNE(n_components=2, perplexity=15, learning_rate=10)
        result = tsne.fit_transform(self.output.cpu().numpy())

        if category is None:
            vis_x = result[:, 0]
            vis_y = result[:, 1]
            plt.scatter(vis_x, vis_y, cmap=plt.cm.get_cmap("jet", 10), marker='.')
            plt.colorbar(ticks=range(10))
            plt.clim(-0.5, 9.5)
            plt.show()
        else:
            categories = category
            for i in range(len(categories)):
                category_indices = categories[i]
                plt.scatter(result[category_indices, 0], result[category_indices, 1], color=colors[i],
                            label=f'Category {i}')

            # Add a legend to the plot
            plt.legend()
            plt.show()

    def get_objects(self, index, dims, min_patch_num=8):
        idx = np.unravel_index(index, dims)
        bipartition = np.zeros(dims)
        bipartition[idx[0], idx[1]] = 1

        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        objects, num_objects = ndimage.label(bipartition, structure=s)

        object_list = []
        plt_list = []
        for idx in range(num_objects):
            cc = np.array(np.where(objects == idx + 1))
            if cc.shape[1] < min_patch_num:
                continue
            object_list.append(cc)
            mask = np.zeros(dims)
            mask[cc[0], cc[1]] = 1
            plt_list.append(mask)
        return object_list, plt_list

    def calculate_ncut(self, fore, back):
        cut_fore_back = 0
        asso_fore = 0
        asso_back = 0
        nodes = np.concatenate([fore, back])
        for i in range(fore.shape[0]):
            idx = fore[i]
            asso_fore += np.sum(self.correspend[idx][nodes])
            cut_fore_back += np.sum(self.correspend[idx][back])
        for i in range(back.shape[0]):
            idx = back[i]
            asso_back += np.sum(self.correspend[idx][nodes])

        return cut_fore_back / asso_fore + cut_fore_back / asso_back

    def calculate_purity(self, np_list):
        idx_list = np.ravel_multi_index(np_list, self.dims)
        count = np.sum(self.pseudo_labels[idx_list])
        purity = count / len(idx_list)
        return purity

    def norm_cut(self, idx_list, k=1, tau=0.2, eps=1e-5, visualize=False,
                 no_binary_graph=False, use_eigenvec=False, use_ncut_value=True):
        A = self.correspend[np.ix_(idx_list, idx_list)]
        D = self.di_diag[np.ix_(idx_list, idx_list)]

        # Print second and third smallest eigenvector
        eigenvalue, eigenvectors = eigh(D - A, D, subset_by_index=[k, k])
        # if use_eigenvec:
        #     if eigenvalue > 0.8:
        #         count = np.sum(self.pseudo_labels[idx_list])
        #         purity = count / len(idx_list)
        #         if purity < 0.2:
        #             return None, None, None, None, None, None

        eigenvec = np.copy(eigenvectors[:, 0])

        # Using average point to compute bipartition
        second_smallest_vec = eigenvectors[:, 0]
        avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
        bipartition = second_smallest_vec > avg

        seed = np.argmax(np.abs(second_smallest_vec))

        if bipartition[seed] != 1:
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)

        # only keep indices （这里只进行了二分）
        fore = idx_list[bipartition]
        back = idx_list[~bipartition]

        # verse to patch size
        # fore_objects，back_objects: fore_objects，back_objects的列表，将刚刚的二分图根据连通性进行选择
        # fore_objects, fore_plt = self.get_objects(fore, self.dims, min_patch_num=12)
        # back_objects, back_plt = self.get_objects(back, self.dims, min_patch_num=20)
        fore_objects, fore_plt = self.get_objects(fore, self.dims, min_patch_num=4)
        back_objects, back_plt = self.get_objects(back, self.dims, min_patch_num=4)

        if visualize:
            for object in fore_plt:
                plt.imshow(object)
                plt.show()
            for object in back_plt:
                plt.imshow(object)
                plt.show()

        return fore_objects, back_objects, fore, back, fore_plt, back_plt

    def get_objects_coordinate(self, obj):
        assert obj.shape[0] == 2
        y_c, x_c = obj
        ymin = int(min(y_c) * self.patch_size * self.rate)
        ymax = int(max(y_c) * (self.patch_size+1) * self.rate)
        xmin = int(min(x_c) * self.patch_size * self.rate)
        xmax = int(max(x_c) * (self.patch_size+1) * self.rate)
        return xmin, ymin, xmax, ymax

    def get_algebraic_connectivity(self, obj):
        idx_list = np.ravel_multi_index(obj, self.dims)
        A = self.correspend[np.ix_(idx_list, idx_list)]
        D = self.di_diag[np.ix_(idx_list, idx_list)]
        k = 1
        # Print second and third smallest eigenvector
        eigenvalue, _ = eigh(D - A, D, subset_by_index=[k, k])
        return eigenvalue

    def get_boxes(self, objs):
        boxes = []
        if objs:
            for obj in objs:
                box = self.get_objects_coordinate(obj)
                boxes.append(box)
        return boxes


    def ncut_by_threshold(self, visualize=False, timing=True):
        start_time = time.time()
        last_time = time.time()

        indices = np.arange(self.h_featmap * self.w_featmap)
        objects = []
        min_patch_num = 80
        objects.append(indices)

        num = 0
        while len(objects) != 0:
            fore_objects_boxes = []
            back_objects_boxes = []
            fore_objects, back_objects, fore, back, fore_plt, back_plt = self.norm_cut(objects[0], visualize=visualize, use_eigenvec=True,
                                                                   use_ncut_value=False)
            num +=1
            print(num, "iters:")

            if fore_objects:
                for fo in fore_objects:
                    fore_box = self.get_objects_coordinate(fo)
                    fore_objects_boxes.append(fore_box)
            print(fore_objects_boxes)
            if back_objects:
                for bo in back_objects:
                    back_box = self.get_objects_coordinate(bo)
                    back_objects_boxes.append(back_box)
            print(back_objects_boxes)


            if fore_objects:
                for fo in fore_objects:
                    alg_con = self.get_algebraic_connectivity(fo)
                    if alg_con > 0.8:
                        continue
                    objects.append(np.ravel_multi_index(fo, self.dims))

            if back_objects:
                for bo in back_objects:
                    alg_con = self.get_algebraic_connectivity(bo)
                    if alg_con > 0.8:
                        continue
                    objects.append(np.ravel_multi_index(bo, self.dims))

            if len(fore_objects_boxes) > 0:
                print(len(fore_objects_boxes))
                show_img_and_box(self.ori_img, fore_objects_boxes, title=str(fore_objects_boxes))
            if len(back_objects_boxes) > 0:
                print(len(back_objects_boxes))
                show_img_and_box(self.ori_img, back_objects_boxes, title=str(back_objects_boxes))

            del objects[0]

        now_time = time.time()
        print("Image input:")
        print("total time: ", now_time - start_time)
        print("module time: ", now_time - last_time)
        last_time = time.time()

    def kcut(self):
        indices = np.arange(self.h_featmap * self.w_featmap)
        objects = []
        boxes = []
        maskes = []
        objects.append(indices)

        while len(objects) != 0:
            fore_objects_boxes = []
            back_objects_boxes = []
            fore_objects, back_objects, fore, back, fore_plt, back_plt = self.norm_cut(objects[0], visualize=False,
                                                                                       use_eigenvec=True,
                                                                                       use_ncut_value=False)

            if fore_objects:
                for fo in fore_objects:
                    fore_box = self.get_objects_coordinate(fo)
                    fore_objects_boxes.append(fore_box)
                    boxes.append(fore_box)
                for fp in fore_plt:
                    maskes.append(fp)
            if back_objects:
                for bo in back_objects:
                    back_box = self.get_objects_coordinate(bo)
                    back_objects_boxes.append(back_box)
                    boxes.append(back_box)
                for bp in back_plt:
                    maskes.append(bp)

            if fore_objects:
                for fo in fore_objects:
                    alg_con = self.get_algebraic_connectivity(fo)
                    if alg_con > 0.8:
                        continue
                    objects.append(np.ravel_multi_index(fo, self.dims))

            if back_objects:
                for bo in back_objects:
                    alg_con = self.get_algebraic_connectivity(bo)
                    if alg_con > 0.8:
                        continue
                    objects.append(np.ravel_multi_index(bo, self.dims))
            del objects[0]

        return boxes, maskes

    def save_prm_and_box_list(self, dataset, prm_list, box_list, fsave, mask_save_path, img_path, save_info=True):
        if dataset == "Ins160" or dataset == "Ins335" or dataset == "CUHK-SYSU" or dataset == "gdd" or dataset == "coco_test2017" or dataset == "coco_val2017" or dataset == "VOC_test2007" or dataset == "oxford" or dataset == "paris" or dataset == "holidays":
            name = img_path.split('.')[0].split('/')[-1]
        elif dataset == "instre" or dataset == "test_instre":
            name = "/".join(img_path.split('.')[0].split('/')[-3:]).replace("/", "_")
        else:
            raise NotImplementedError
        prm_np = np.array(prm_list).astype(np.uint8)
        if save_info:
            file_name = mask_save_path + "/" + name + ".npy"
            np.save(file_name, prm_np)
        else:
            print(prm_np)

        for i in range(len(box_list)):
            info = img_path + ">" + str(i).rjust(4, '0') + " " + " ".join(str(num) for num in box_list[i]) + "\n"
            if save_info:
                fsave.write(info)
            else:
                print(info)

    def estimate_ellipse(self, y, x, img_shape, peak_response_map=None):
        """
        :param y: np.array
        :param x: np.array
        :param img_shape: (h, w)
        :return:
        """
        above_points = np.stack([y, x], axis=-1)
        above_points_mu = np.mean(above_points, axis=0)
        above_points_minus_mu = above_points - above_points_mu
        cov = np.matmul(above_points_minus_mu.transpose(), above_points_minus_mu)
        cov = cov / np.shape(above_points_minus_mu)[0]
        values, vectors = scipy.linalg.eig(cov)
        values = np.eye(values.shape[0]) * values
        # values = 2.5 * np.sqrt(values)
        values = 2 * np.sqrt(values)
        angle = math.atan2(cov[1, 0] + cov[0, 1], cov[1, 1] - cov[0, 0]) / 2 / math.pi * 180
        minor_axis, major_axis = np.sort(values.flatten())[-2:]
        angle_radius = angle / 180.0 * np.pi
        ux = major_axis * np.cos(angle_radius)
        uy = major_axis * np.sin(angle_radius)
        vx = minor_axis * np.cos(angle_radius + np.pi / 2.)
        vy = minor_axis * np.sin(angle_radius + np.pi / 2.)
        half_width = np.sqrt(ux ** 2 + vx ** 2)
        half_height = np.sqrt(uy ** 2 + vy ** 2)
        y_min, y_max = np.clip(
            np.array([-half_height, half_height], dtype=np.int32) + int(above_points_mu[0]), 0, img_shape[0] - 1)
        x_min, x_max = np.clip(
            np.array([-half_width, half_width], dtype=np.int32) + int(above_points_mu[1]), 0, img_shape[1] - 1)
        return np.array([x_min, y_min, x_max, y_max])

    def get_mask_from_prm(self, prm):
        mask = prm.numpy()
        mask[mask > 0] = 1
        return mask

    def display_multi_plt(self, list, num_cols, is_jet=False, boxes=None):
        num_plots = len(list)
        num_rows = (num_plots + num_cols - 1) // num_cols

        # 创建figure对象和子图数组
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 6))
        gs = GridSpec(num_rows, num_cols, figure=fig)

        ax = axes.flatten()
        if boxes is None:
            for i in range(num_cols * num_rows):
                if i < num_plots:
                    if is_jet:
                        ax[i].imshow(list[i], cmap=plt.cm.jet)
                    else:
                        colors = [(178 / 255, 223 / 255, 238 / 255, 1),
                                  (205 / 255, 102 / 255, 0 / 255, 1)]  # 淡蓝色和黄色，以RGB格式表示
                        n_bins = [0, 0.5, 1.0]  # 数据值的分界点
                        cmap_name = 'custom_colormap'
                        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=len(n_bins) - 1)

                        ax[i].imshow(list[i], cmap=custom_cmap)
                        # ax[i].imshow(list[i], cmap='YlGn')
                ax[i].axis('off')
        else:
            assert len(boxes) == num_plots, "Wrong boxes list input!"
            for i in range(num_cols * num_rows):
                if i < num_plots:
                    if is_jet:
                        ax[i].imshow(list[i], cmap=plt.cm.jet)
                        print(boxes[i])
                        ax[i].add_patch(
                            plt.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2] - boxes[i][0],
                                          boxes[i][3] - boxes[i][1], fill=False,
                                          edgecolor='r', linewidth=1))
                    else:
                        ax[i].imshow(list[i])
                        ax[i].add_patch(
                            plt.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2] - boxes[i][0],
                                          boxes[i][3] - boxes[i][1], fill=False,
                                          edgecolor='r', linewidth=1))
                ax[i].axis('off')
        gs.update(wspace=0.2, hspace=0.2)
        plt.show()

    def backprop(self, img, model, w_featmap, h_featmap, patch_size, object_list, output, attentions, rate,
                 visualize=False):
        # 13, 20
        img.requires_grad_()
        # output, attentions = model.get_last_output_and_selfattention(img)
        output = output[:, 1:].squeeze(0).sum(-1)
        # 获得num_heads 和 cls（对于全局）的attention
        nh = attentions.shape[1]  # number of head
        # we keep only the output patch attention
        mean_attention = attentions.sum(0)

        prm_list = []
        box_list = []
        result_list = []
        mask_list = []
        for obj in object_list:
            if len(obj) == 0:
                continue
            grad = torch.zeros_like(output)
            if img.grad is not None:
                img.grad.zero_()
            for i in range(obj.shape[1]):
                h, w = obj[:, i]
                # grad[h * w_featmap + w] = 1
                grad[h * w_featmap + w] = mean_attention[h * w_featmap + w]
            output.requires_grad_(True)
            output.backward(grad, retain_graph=True)
            prm = img.grad.detach().sum(0).clone().clamp(min=0).squeeze().cpu()
            mask = self.get_mask_from_prm(prm)
            # print(prm.shape)

            prm_points = np.nonzero(prm)
            # print(prm_points)
            x = prm_points[:, 0]
            y = prm_points[:, 1]
            box = self.estimate_ellipse(x, y, (prm.shape[0], prm.shape[1]), prm).tolist()

            box_list.append(box)
            prm_list.append(prm)
            mask_list.append(mask)
            result = [int(x * rate) for x in box]
            result_list.append(result)

        if visualize:
            self.display_multi_plt(mask_list, num_cols=5, is_jet=True)
        return result_list, mask_list

    def extract_dataset(self, dataset=None, save_path=None):
        assert dataset != None
        assert save_path != None

        path = save_path + "/" + dataset + "/mask"
        if not os.path.exists(path):
            os.makedirs(path)

        if dataset == "Ins160":
            file_path = "/home/ybmiao/yb_data/path_Instance-160.txt"
        elif dataset == "Ins335":
            file_path = "/home/ybmiao/yb_data/path_Instance-335.txt"
        elif dataset == "instre":
            file_path = "/home/ybmiao/yb_data/path_INSTRE.txt"
        elif dataset == "test_instre":
            file_path = "/media/media01/qysun/data/INSTRE/small/small_BracelonaFC.txt"
        elif dataset == "CUHK-SYSU":
            file_path = "/media/media01/qysun/data/CUHK-SYSU/INS/ref-for-myb.txt"
        elif dataset == "gdd":
            # file_path = "/home/ybmiao/data/gdd/gdd_path.txt"
            file_path = "/home/ybmiao/code/EMB/pruning_algorithm/output/samples/samples_path.txt"
        elif dataset == "coco_val2017":
            file_path = "/media/media01/qysun/data/coco/coco_val2017_path.txt"
        elif dataset == "VOC_test2007":
            file_path = "/media/media01/qysun/data/VOC/VOC2007_test_path.txt"
        elif dataset == "coco_test2017":
            file_path = "/home/ybmiao/data/coco/coco_test2017_path.txt"
        elif dataset == "oxford":
            file_path = "/media/media01/qysun/data/oxford/image_path.txt"
        elif dataset == "holidays":
            file_path = "/media/media01/qysun/data/holidays/Holidays_path.txt"
        elif dataset == "paris":
            file_path = "/media/media01/qysun/data/paris/image_path.txt"
        else:
            raise ValueError("Wrong dataset input!")

        frd = open(file_path, 'r')
        fsave = open(save_path + "/" + dataset + "/box_ref.txt", "a")
        mask_save_path = os.path.join(save_path, dataset, 'mask')
        lines = frd.readlines()
        for idx, line in enumerate(tqdm(lines)):
            img_path = line.strip()
            # img_path = "/media/media01/qysun/data/oxford/oxbuild_images/cornmarket_000047.jpg"
            # if "09ConfusionVideo00004" not in img_path:
            #     continue
            # if idx < 4794:
            #     continue
            # print(img_path)

            # get input tensor
            self.update_init_info(img_path=img_path)

            # use_mbp = False
            use_mbp = True
            if use_mbp:
                _, maskes = self.kcut()
                objs = []
                for mask in maskes:
                    ones_coordinates = np.argwhere(mask == 1)
                    x_coordinates = ones_coordinates[:, 0]
                    y_coordinates = ones_coordinates[:, 1]
                    # obj = np.array([x_coordinates, y_coordinates])[np.newaxis, :, :]
                    obj = np.array([x_coordinates, y_coordinates])
                    objs.append(obj)

                boxes = []
                prms = []
                # for mask in maskes:
                #     if len(mask) != 0:
                #         box, prm = self.backprop(self.img, self.model, self.w_featmap, self.h_featmap, self.patch_size, mask, self.output,
                #                                   self.attens, self.rate, visualize=False)
                #         boxes += box
                #         prms += prm


                box, prm = self.backprop(self.img, self.model, self.w_featmap, self.h_featmap, self.patch_size, objs, self.output,
                                          self.attens, self.rate, visualize=False)
                # print(box)
                # print(prm[0].shape)
                # print(self.ori_img.shape)
                # exit()

                boxes += box
                prms += prm
                self.save_prm_and_box_list(dataset, prms, boxes, fsave, mask_save_path, img_path, save_info=True)
                self.bp._grad_zero()
                del maskes
                del self.output, self.attens
            else:
                boxes, maskes = self.kcut()
                # print(len(boxes))
                # print(len(maskes))
                # exit()
                self.save_prm_and_box_list(dataset, maskes, boxes, fsave, mask_save_path, img_path)
                del maskes
                del self.output, self.attens

            
        
        
    
if __name__ == "__main__":
    # img_path = '/home/ybmiao/data/INSTRE/INSTRE-S1/47b_wierd_fish/022.jpg'
    # img_path = '/home/ybmiao/data/INSTRE/INSTRE-S1/16b_taiwan101/047.jpg'
    # img_path = '/home/ybmiao/yb_data/test/test.jpg'
    # img_path = '/home/ybmiao/yb_data/test/iris-3.jpeg'
    # img_path = '/home/ybmiao/data/INSTRE/INSTRE-S1/47b_wierd_fish/022.jpg'
    # img_path = '/home/ybmiao/data/INSTRE/INSTRE-M/50/107.jpg' # 也不错
    # self.img_path = '/home/ybmiao/data/INSTRE/INSTRE-M/20/013.jpg'
    # self.img_path = '/home/ybmiao/data/INSTRE/INSTRE-M/20/070.jpg'
    # self.img_path = '/home/ybmiao/data/INSTRE/INSTRE-S1/01a_canada_book/007.jpg'  # 好结果展示
    # self.img_path = '/home/ybmiao/data/INSTRE/INSTRE-S1/16b_taiwan101/047.jpg'
    # self.img_path = '/media/media01/qysun/data/INSTRE/INSTRE-M/01/001.jpg'
    # img_path = '/media/media01/qysun/data/INSTRE/INSTRE-M/01/004.jpg'
    # img_path = '/media/media01/qysun/data/INSTRE/INSTRE-M/01/006.jpg' # 已经非常好了
    # img_path = '/media/media01/qysun/data/INSTRE/INSTRE-M/01/049.jpg' # 就这张图了
    # img_path = '/media/media01/qysun/data/INSTRE/INSTRE-M/01/035.jpg' # 就这张图了
    # self.img_path = '/media/media01/qysun/data/INSTRE/INSTRE-M/01/050.jpg' # 就这张图了

    # img_path = '/home/ybmiao/data/Instance-335/Images/13ZoomingCameraVideo00024/13ZoomingCameraVideo00024_00000121.jpg'
    # img_path = '/media/media01/qysun/data/Instance-160/Images/09ConfusionVideo00004/09ConfusionVideo00004_00000091.jpg'
    # img_path = '/home/ybmiao/data/Instance-335/Images/04TransparencyVideo00001/04TransparencyVideo00001_00000003.jpg' # 两头牛
    # img_path = '/home/ybmiao/data/Instance-335/Images/01LightVideo00002/01LightVideo00002_00000001.jpg'  # 唱歌
    # img_path = '/home/ybmiao/data/Instance-335/Images/Human9/Human9_0036.jpg'  # 街道+男人
    # img_path = '/home/ybmiao/data/Instance-335/Images/Woman/Woman_0446.jpg'  # 俩车一女人
    # img_path = '/home/ybmiao/data/Instance-335/Images/01LightVideo00001/01LightVideo00001_00000001.jpg'  # 滑雪女孩
    # img_path = '/home/ybmiao/data/Instance-335/Images/CarScale/CarScale_0121.jpg'  # 房子+suv
    # img_path = '/home/ybmiao/data/Instance-335/Images/11OcclusionVideo00013/11OcclusionVideo00013_00000145.jpg' # 弹钢琴的女人
    # img_path = '/home/ybmiao/data/Instance-335/Images/Car24/Car24_1826.jpg'  # 街道汽车
    # img_path = '/home/ybmiao/data/Instance-335/Images/05ShapeVideo00008/05ShapeVideo00008_00000016.jpg' # 单双杠

    # img_path = '/media/media01/qysun/code/object_location/sqy_output/Mimage.jpg'
    # img_path = '/home/ybmiao/data/coco/test2017/000000449501.jpg'
    img_path = '/home/ybmiao/code/EMB/pruning_algorithm/output/samples/DJI_0075_1.jpg'

    model_name = "dino"
    patch_size = 8

    # model_name = "dinov2"
    # patch_size = 14

    # model_name = "pretrained"
    # patch_size = 16

    # model_name = "clip"
    # patch_size = 16

    graphv = Graph_Visualization(img_path, model_name, patch_size)
    # graphv.tsne_visualize()
    # graphv.feat_ncut(visualize=True)
    # graphv.ncut_by_threshold(visualize=True)
    # graphv.ncut_by_threshold(visualize=False)
    # graphv.feat_ncut(tau=0.3)
    # graphv.extract_dataset(dataset="instre", save_path="/home/ybmiao/output/kcut/12_07")
    # graphv.extract_dataset(dataset="Ins160", save_path="/home/ybmiao/output/kcut/3_22/thes_06")
    # graphv.extract_dataset(dataset="holidays", save_path="/home/ybmiao/output/kcut/3_21")
    # graphv.extract_dataset(dataset="gdd", save_path="/home/ybmiao/output/kcut/12_07")
    # graphv.extract_dataset(dataset="VOC_test2007", save_path="/home/ybmiao/output/kcut/1_10")
    # graphv.extract_dataset(dataset="oxford", save_path="/home/ybmiao/output/kcut/5_17")
    graphv.extract_dataset(dataset="holidays", save_path="/home/ybmiao/output/kcut/5_17")
    # graphv.extract_dataset(dataset="paris", save_path="/home/ybmiao/output/kcut/3_21")
    # graphv.extract_dataset(dataset="coco_test2017", save_path="/home/ybmiao/output/kcut/12_21")
    # graphv.extract_dataset(dataset="voc07", save_path="/home/ybmiao/output/kcut/12_21")