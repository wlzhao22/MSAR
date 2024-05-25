import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

dict = torch.load("/media/media01/ybmiao/output/Exhaust/dss/coco_val2017/eigs/laplacian/000000495146.pth")
print(dict['eigenvectors'].shape)


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
