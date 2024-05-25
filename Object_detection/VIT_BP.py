from types import MethodType

import matplotlib.pyplot as plt
from torchvision import transforms as T
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from functions import pr_conv2d, pr_attention
import vision_transformer as vits



class VIT_Backprop(nn.Sequential):
    def __init__(self, *args, **kargs):
        super().__init__(*args)
        # print(super())
        # print(*args)

        self.inferencing = False
        # use global average pooling to aggregate responses if peak stimulation is disabled
        self.enable_peak_stimulation = kargs.get('enable_peak_stimulation', True)
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = kargs.get('enable_peak_backprop', True)
        # window size for peak finding
        self.win_size = kargs.get('win_size', 3)
        # sub-pixel peak finding
        self.sub_pixel_locating_factor = kargs.get('sub_pixel_locating_factor', 1)
        # peak filtering
        self.filter_type = kargs.get('filter_type', 'median')

    # def _patch(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d):
    #             module._original_forward = module.forward
    #             module.forward = MethodType(pr_conv2d, module)
    #
    # def _recover(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
    #             module.forward = module._original_forward

    def _patch(self):
        for module in self.modules():
            if module.__class__.__name__ == 'Attention':
                module._original_forward = module.forward
                module.forward = MethodType(pr_attention, module)

    def _recover(self):
        for module in self.modules():
            if module.__class__.__name__ == 'Attention' and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    def _grad_zero(self):
        for module in self.modules():
            if module.__class__.__name__ == 'Attention' and hasattr(module, '_original_forward'):
                module.zero_grad()

    def show_info(self):
        for module in self.modules():
            print(module)
            print(module.__class__.__name__)


# backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
# bp = VIT_Backprop(backbone)
#
# # bp.show_info()
# bp._patch()
# # bp._recover()
#
#
# img_path = "/home/ybmiao/data/INSTRE/INSTRE-M/20/013.jpg"
# img = Image.open(img_path,"r")
# trans = T.Compose([T.ToTensor(),
#                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# input = trans(img).unsqueeze(0)
# input.requires_grad_()
# output, attens = backbone.get_last_output_and_selfattention(input)
#
# print(input.shape)
# print(output.shape)
# print(attens.shape)
#
#
# output = output[:, 1:].squeeze(0).sum(-1)
# # mean_attention = attens.sum(0)
# h,w = 10,8
# grad = torch.zeros_like(output)
# grad[80] = 1
# grad[81] = 1
# grad[82] = 1
# grad[83] = 1
# if input.grad is not None:
#     input.grad.zero_()
# output.requires_grad_(True)
# output.backward(grad, retain_graph=True)
# prm = input.grad.detach().sum(1).clone().clamp(min=0).squeeze().cpu()
# prm[prm > 0] = 1
# plt.imshow(prm)
# plt.show()