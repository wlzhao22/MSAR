from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.autograd import Function


class PreHook(Function):

    @staticmethod
    def forward(ctx, input, offset):
        ctx.save_for_backward(input, offset)
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input, offset = ctx.saved_variables
        return (input - offset) * grad_output, None


# class PostHook(Function):
#
#     @staticmethod
#     def forward(ctx, input, norm_factor):
#         ctx.save_for_backward(norm_factor)
#         return input.clone()
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         norm_factor, = ctx.saved_variables
#         eps = 1e-10
#         zero_mask = norm_factor < eps
#         grad_input = grad_output / (torch.abs(norm_factor) + eps)
#         grad_input[zero_mask.detach()] = 0
#         return None, grad_input

class PostHook(Function):

    @staticmethod
    def forward(ctx, input, norm_factor):
        ctx.save_for_backward(norm_factor)
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        norm_factor, = ctx.saved_variables
        # eps = 1e-10
        eps = 1e-10
        zero_mask = norm_factor < eps
        # grad_input = grad_output / (torch.abs(norm_factor) + eps)
        grad_input = grad_output / (torch.abs(norm_factor) + 1e-10)
        grad_input[zero_mask.detach()] = 0
        return None, grad_input


def pr_conv2d(self, input):
    offset = input.min().detach()
    input = PreHook.apply(input, offset)
    resp = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups).detach()
    pos_weight = F.relu(self.weight).detach()
    norm_factor = F.conv2d(input - offset, pos_weight, None, self.stride, self.padding, self.dilation, self.groups)
    output = PostHook.apply(resp, norm_factor)
    return output



def pr_attention(self, input):
    offset = input.min().detach()
    input = PreHook.apply(input, offset)

    B, N, C = input.shape
    qkv = self.qkv(input).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    norm_factor = torch.ones_like(x)
    output = PostHook.apply(x, norm_factor)

    return output, attn