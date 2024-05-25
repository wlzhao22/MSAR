import os

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

def show_img_and_box(img, boxes, title=None, axis=False):
    colors = ['r', 'g', 'b', 'y', 'm']
    fig, ax = plt.subplots()
    ax.imshow(img)
    print(img.shape)
    for i in range(len(boxes)):
        ax.add_patch(
            plt.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2] - boxes[i][0],
                          boxes[i][3] - boxes[i][1], fill=False,
                          edgecolor=colors[i % 5], linewidth=3))
    print(title)
    if title is not None:
        print(title)
        plt.title(title)
    if not axis:
        plt.axis("off")
    plt.show()

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
    elif dataset == "coco":
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

dst_root = "/home/ybmiao/output/Exhaust/dss/voc_test2007"
src_root = "/home/ybmiao/output/Exhaust/dss/voc_test2007/eigs/laplacian"
img_root = "/media/media01/qysun/data/VOC/VOC07_test/VOCdevkit/VOC2007/JPEGImages/"
num = 0
dst_f = open(os.path.join(dst_root,"box_ref.txt"),"w")
for path in tqdm(os.listdir(src_root)):
    file_path = os.path.join(src_root, path)
    img_path  = os.path.join(img_root, path.split(".")[0]+".jpg")
    state_dict = torch.load(file_path)
    result = state_dict['eigenvectors']

    img = Image.open(img_path, "r")
    arr_img = np.asarray(img.convert('RGB')).astype(np.uint8)

    # P = 8
    P = 16
    H, W, C = arr_img.shape
    H_patch, W_patch = H // P, W // P

    plt_list = []
    box_list = []
    for i in range(result.shape[0]-1):
        tensor = result[i+1].view(H_patch, W_patch).cpu().numpy()

        patch_mask = tensor > 0

        if 0.5 < np.mean(patch_mask).item() < 1.0:
            patch_mask = (1 - patch_mask).astype(np.uint8)
        elif np.sum(patch_mask).item() == 0:  # nothing detected at all, so cover the entire image
            patch_mask = (1 - patch_mask).astype(np.uint8)

        #
        # plt.imshow(patch_mask)
        # plt.show()

        plt_list.append(patch_mask)

        xmin, ymin, xmax, ymax = get_largest_cc_box(patch_mask)
        box_up_sampled = [xmin*P, ymin*P, xmax*(P+1), ymax*(P+1)]
        # print(box_up_sampled)
        box_list.append(box_up_sampled)

    # show_img_and_box(arr_img, box_list)

    save_prm_and_box_list(dataset="coco",prm_list=plt_list,box_list=box_list,fsave=dst_f,mask_save_path=os.path.join(dst_root,"mask"), img_path=img_path)


exit()

first_tensor = dict['eigenvectors'][1]
print(first_tensor)
first_tensor = first_tensor.view(60,80)
plt.imshow(first_tensor)
plt.show()

img_path = "/home/ybmiao/data/coco/images/val2017/000000495146.jpg"
img = Image.open(img_path, "r")
plt.imshow(img)
plt.show()

np_arr = np.load("/media/media01/ybmiao/output/EMB/7_22/Ins160/mask/Gym_0261.npy")

print(np_arr.shape)
