import os
import glob

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

def calculate_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    # 计算相交区域的宽度和高度
    intersection_width = min(xmax1, xmax2) - max(xmin1, xmin2)
    intersection_height = min(ymax1, ymax2) - max(ymin1, ymin2)

    # 判断是否相交
    if intersection_width <= 0 or intersection_height <= 0:
        intersection_area = 0
    else:
        intersection_area = intersection_width * intersection_height

    # 计算并集区域的宽度和高度
    union_width = max(xmax1, xmax2) - min(xmin1, xmin2)
    union_height = max(ymax1, ymax2) - min(ymin1, ymin2)

    # 计算 IoU
    iou = intersection_area / (union_width * union_height)

    return iou


def traverse_txt_files(folder_path):
    txt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                txt_files.append(file_path)
    return txt_files


def get_gt_box_dict(dataset):
    gt_box_dict = {}
    gt_path = None
    if dataset == "Ins160":
        gt_path = "/media/media01/qysun/data/Instance-160/bbox"
        txt_files = traverse_txt_files(gt_path)
    elif dataset == "Ins335":
        gt_path = "/home/ybmiao/data/Instance-335/merge_bbox"
        txt_files = traverse_txt_files(gt_path)
    elif dataset == "instre":
        gt_paths = ["/media/media01/qysun/data/INSTRE/INSTRE-S1", "/media/media01/qysun/data/INSTRE/INSTRE-S2", "/media/media01/qysun/data/INSTRE/INSTRE-M"]
        txt_files = []
        for gt_path in gt_paths:
            files = traverse_txt_files(gt_path)
            txt_files.extend(files)
    else:
        raise ValueError("Wrong ref path input!")
    print(len(txt_files))
    # exit()

    if dataset == "Ins160" or dataset == "Ins335":
        for txt in tqdm(txt_files):
            f =  open(txt, "r")
            line = f.readline().strip()
            while line:
                info = line.split()
                box = list(map(int, info[1:]))
                box[2] += box[0]
                box[3] += box[1]
                ll = [box]  # 为了和INSTRE-M中多物体保持一致，使用二维列表
                gt_box_dict[info[0]] = ll
                line = f.readline().strip()
            f.close()
    elif dataset == "instre" :
        for txt in tqdm(txt_files):
            f = open(txt, "r")
            ll = []
            line = f.readline().strip()
            while line:
                info = line.split()
                box = list(map(int, info))
                box[2] += box[0]
                box[3] += box[1]
                ll.append(box)
                line = f.readline().strip()
            name = "/".join(txt.split('.')[0].split('/')[-3:])
            gt_box_dict[name] = ll
            f.close()
    else:
        raise ValueError("Wrong ref path input!")

    return gt_box_dict

def get_ref_box_dict(ref_path, dataset):
    ref_box_dict = {}
    f = open(ref_path, "r")
    line = f.readline().strip()
    cnt = 0
    box_list = []
    old_name = None
    while line:
        info = line.split()
        if dataset == "Ins160" or dataset == "Ins335":
            new_name = info[0].split('.')[0].split('/')[-1]
        elif dataset == "instre":
            new_name = "/".join(info[0].split('.')[0].split('/')[-3:])
        else:
            raise ValueError("Wrong ref path input!")

        if old_name != new_name and cnt != 0:
            ref_box_dict[old_name] = box_list
            box_list = []

        box = list(map(int, info[1:]))
        box_list.append(box)

        old_name = new_name
        line = f.readline().strip()
        cnt += 1

    f.close()
    ref_box_dict[old_name] = box_list
    return ref_box_dict

def load_real_ref(real_ref, dataset):
    ref_list = []
    path_list = []
    f = open(real_ref, 'r')
    line = f.readline().strip()
    while line:
        if dataset == "Ins160" or dataset == "Ins335":
            name = line.split('.')[0].split('/')[-1]
        elif dataset == "instre":
            name = "/".join(line.split('.')[0].split('/')[-3:])
        else:
            raise ValueError("Wrong ref path input!")
        ref_list.append(name)
        path_list.append(line)
        line = f.readline().strip()

    print(len(ref_list))
    return ref_list, path_list

def calculate_miou(gt_box, ref_box, real_ref, dataset, visual=False):
    ref_list, path_list = load_real_ref(real_ref, dataset)
    mious = 0.0
    for idx, (ref, path) in tqdm(enumerate(zip(ref_list, path_list))):
        miou = 0.0

        if ref not in ref_box:
            mious += miou
            continue

        for rb in ref_box[ref]:
            for gb in gt_box[ref]: # 如果有两个查询，则计算iou更大的那个gtbox作为iou
                result = calculate_iou(rb,gb)
                if miou < result:
                    miou = result
        # if len(ref_box[ref]) != 0:
        #     miou /= len(ref_box[ref])
        mious += miou

        if idx < 100:
            print(path, "miou:", miou)

        if visual:
            # if "04TransparencyVideo00017" not in path:
            # if "INSTRE-M/29/" not in path:
            if "INSTRE-S1/47b_wierd_fish/" not in path:
            # if "CarScale" not in path:
                continue
            img = Image.open(path)
            plt.imshow(img)
            # print(img.size)
            # print(gt_box[ref])
            # print(ref_box[ref])
            # continue

            for site in gt_box[ref]:
                plt.gca().add_patch(
                    plt.Rectangle((site[0], site[1]), site[2] - site[0],
                                  site[3] - site[1], fill=False,
                                  edgecolor='r', linewidth=1))
            for site in ref_box[ref]:
                plt.gca().add_patch(
                    plt.Rectangle((site[0], site[1]), site[2] - site[0],
                                  site[3] - site[1], fill=False,
                                  edgecolor='g', linewidth=1))

            plt.show()
            # if idx > 5:
            #     exit()
            # print(miou)


    # print(len(ref_list))
    mious /= len(ref_list)
    return mious

def main(ref_path):
    dataset = None
    if "160" in ref_path:
        dataset = "Ins160"
        real_ref = "/home/ybmiao/yb_data/path_Instance-160.txt"
    elif "335" in ref_path:
        dataset = "Ins335"
        real_ref = "/home/ybmiao/yb_data/path_Instance-335.txt"
    elif "instre" or "INSTRE" in ref_path:
        dataset = "instre"
        real_ref = "/home/ybmiao/yb_data/path_INSTRE.txt"
    else:
        raise ValueError("Wrong ref path input!")


    print("Loading Ground Truth Boxes......")
    gt_box_dict = get_gt_box_dict(dataset)
    print(len(gt_box_dict))
    print("Loading Reference Boxes......")
    ref_box_dict = get_ref_box_dict(ref_path, dataset)
    print(len(ref_box_dict))
    print("Calculating mIOU......")
    miou = calculate_miou(gt_box_dict, ref_box_dict, real_ref, dataset)
    # miou = calculate_miou(gt_box_dict, ref_box_dict, real_ref, dataset, visual=True)
    print("mIOU is ", miou)



if __name__ == "__main__":
    ref_path_instre = "/media/media01/ybmiao/output/Exhaust/cutler/instre/box_ref.txt"

    main(ref_path_instre)
    # main(ref_path)