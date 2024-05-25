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
import datetime
from yb_utils import show_img_and_box


class Graph_Visualization():
    def __init__(self, img_path, model_name, patch_size):
        self.ori_img = np.asarray(Image.open(img_path).convert('RGB')).astype(np.uint8)
        self.img, self.rate = self.get_input_var(img_path, show_img=True)
        self.patch_size = patch_size
        self.h_featmap = self.img.shape[-2] // self.patch_size
        self.w_featmap = self.img.shape[-1] // self.patch_size
        self.dims = [self.h_featmap, self.w_featmap]
        self.indices = np.arange(self.h_featmap * self.w_featmap)

        self.patch_size = patch_size
        self.gpu = "1"
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        # self.device = torch.device("cuda:" + self.gpu)
        accelerator = Accelerator(mixed_precision="fp16", cpu=False)
        self.device = accelerator.device
        self.model_name = model_name
        self.output = self.get_output(self.img, model_name=self.model_name)
        self.output = F.normalize(self.output, p=2)
        self.correspend = (self.output @ self.output.transpose(1, 0)).cpu().numpy()
        self.tau = 0.2
        self.eps = 1e-5
        self.no_binary_graph = False
        if self.no_binary_graph:
            self.correspend[self.correspend < self.tau] = self.eps
        else:
            self.correspend = self.correspend > self.tau
            self.correspend = np.where(self.correspend.astype(float) == 0, self.eps, self.correspend)
        self.d_i = np.sum(self.correspend, axis=1)
        self.pseudo_labels = self.init_pseudo_labels(threshold=1, visualize=True)
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
        img = transform(img)
        rate = h / img.shape[-2]

        return img, rate

    def get_output(self, img, model_name="dino"):

        img = img.to(self.device)

        # get related parameter
        w, h = img.shape[1] - img.shape[1] % self.patch_size, img.shape[2] - img.shape[2] % self.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        if model_name == "dino":
            output, attens = get_dino_output(img, self.device)
            # output = output[0, 1:, :].detach().cpu().numpy()
            output = output[0, 1:, :].detach()
        elif model_name == "dinov2":
            output = get_dinov2_output(img, self.device)[0][0].detach().cpu().numpy()
        elif model_name == "pretrained":
            output = get_pretrained_output(img, self.device)
        elif model_name == "clip":
            output = get_clip_output(img, self.device)
        else:
            raise NotImplementedError
        return output

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

    def calculate_rank(self, np_list):
        arr_y = np_list[0]
        arr_x = np_list[1]
        length = arr_x.shape[0]
        patches = []
        # print(self.ori_img.shape)
        # exit()
        for i in range(length):
            x,y = arr_x[i],arr_y[i]
            patch = self.img[100:160, 20:40, :]

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
        if use_eigenvec:
            if eigenvalue > 0.8:
                count = np.sum(self.pseudo_labels[idx_list])
                purity = count / len(idx_list)
                if purity < 0.2:
                    return None, None, None, None

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

        ncut_value = self.calculate_ncut(fore, back)
        print("ncut value:", ncut_value)
        if use_ncut_value:
            if ncut_value > 0.7:
                return None, None, None, None

        # verse to patch size
        # fore_objects，back_objects: fore_objects，back_objects的列表，将刚刚的二分图根据连通性进行选择
        fore_objects, fore_plt = self.get_objects(fore, self.dims, min_patch_num=12)
        back_objects, back_plt = self.get_objects(back, self.dims, min_patch_num=20)


        current_time = datetime.datetime.now()
        if visualize:
            for idx, object in enumerate(fore_plt):
                plt.imshow(object)
                plt.title("fore:"+str(ncut_value))
                fig_name = "./img/fore-img-{:%H:%M:%S}-".format(current_time)
                fig_name = fig_name + str(idx) + ".png"
                plt.savefig(fig_name)
                # plt.show()
                
            for object in back_plt:
                plt.imshow(object)
                plt.title("back:"+str(ncut_value))
                current_time = datetime.datetime.now()
                fig_name = "./img/back-img-{:%H:%M:%S}".format(current_time)
                fig_name = fig_name + str(idx) + ".png"
                plt.savefig(fig_name)
                # plt.show()

        return fore_objects, back_objects, fore, back

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
            fore_objects, back_objects, fore, back = self.norm_cut(objects[0], visualize=visualize, use_eigenvec=True,
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
                    # fore_box = self.get_objects_coordinate(fo)
                    alg_con = self.get_algebraic_connectivity(fo)
                    if alg_con > 0.8:
                        continue
                    objects.append(np.ravel_multi_index(fo, self.dims))
            #         fore_objects_boxes.append(fore_box)
            # print(fore_objects_boxes)

            if back_objects:
                for bo in back_objects:
                    # back_box = self.get_objects_coordinate(bo)
                    alg_con = self.get_algebraic_connectivity(bo)
                    if alg_con > 0.8:
                        continue
                    objects.append(np.ravel_multi_index(bo, self.dims))
            #         back_objects_boxes.append(back_box)
            # print(back_objects_boxes)

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

if __name__ == "__main__":
    # img_path = '/home/ybmiao/data/INSTRE/INSTRE-S1/47b_wierd_fish/022.jpg'
    # img_path = '/home/ybmiao/data/INSTRE/INSTRE-S1/16b_taiwan101/047.jpg'
    # img_path = '/home/ybmiao/yb_data/test/test.jpg'
    # img_path = '/home/ybmiao/yb_data/test/iris-3.jpeg'
    # img_path = '/home/ybmiao/data/INSTRE/INSTRE-S1/47b_wierd_fish/022.jpg'
    # img_path = '/home/ybmiao/data/INSTRE/INSTRE-M/50/107.jpg' # 也不错
    # img_path = '/home/ybmiao/data/INSTRE/INSTRE-M/20/013.jpg'
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
    # img_path = '/home/ybmiao/data/Instance-335/Images/others000177/others000177_00000001.jpg'  # 龙的传人
    # img_path = '/home/ybmiao/data/Instance-335/Images/Freeman3/Freeman3_0001.jpg'  # 上课男人
    # img_path = '/home/ybmiao/data/Instance-335/Images/CarScale/CarScale_0121.jpg'  # 房子+suv
    # img_path = '/home/ybmiao/data/Instance-335/Images/11OcclusionVideo00013/11OcclusionVideo00013_00000145.jpg' # 弹钢琴的女人
    # img_path = '/home/ybmiao/data/Instance-335/Images/Car24/Car24_1826.jpg'  # 街道汽车
    # img_path = '/home/ybmiao/data/Instance-335/Images/05ShapeVideo00008/05ShapeVideo00008_00000016.jpg' # 单双杠
    # img_path = "/home/ybmiao/data/Instance-335/Images/Walking/Walking_0001.jpg"
    # img_path = "/home/ybmiao/data/Instance-335/Images/KiteSurf/KiteSurf_0001.jpg"
    # img_path = "/home/ybmiao/data/INSTRE/INSTRE-M/31/020.jpg"

    img_path = "/media/media01/qysun/data/oxford/oxbuild_images/cornmarket_000047.jpg"
    # img_path = '/media/media01/qysun/code/object_location/sqy_output/Mimage.jpg'

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
    graphv.ncut_by_threshold(visualize=True)
    # graphv.ncut_by_threshold(visualize=False)
    # graphv.feat_ncut(tau=0.3)