import vision_transformer as vits
import torch
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
import clip

def get_dino_output(img_var, device):
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    model.eval()
    model.to(device)
    output, attens = model.get_last_output_and_selfattention(img_var)

    return output, attens

def get_dinov2_output(img_var, device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval()
    model.to(device)
    output = model.get_intermediate_layers(img_var, n=1)

    return output

def get_clip_output(img_var, device):
    model, preprocess = clip.load("ViT-B/16", device=device)

    image_features = model.encode_image(img_var)
    print(image_features.shape)
    exit()


def get_pretrained_output(img_var, device):
    # model = torchvision.models.vision_transformer.vit_b_16(pretrained=True)
    #
    # fmap_block = dict()  # 装feature map
    # def forward_hook(module, input, output):
    #     fmap_block['input'] = input
    #     fmap_block['output'] = output
    #
    # model.eval()
    # model.to(device)
    # layer_name = 'encoder'
    # for (name, module) in model.named_modules():
    #     if name == layer_name:
    #         print(name)
    #         module.register_forward_hook(hook=forward_hook)
    # print(fmap_block['output'].shape)
    # exit()
    #
    # output = model.get_intermediate_layers(img_var, n=1)
    #


    model = torchvision.models.vision_transformer.vit_b_16(pretrained=True)
    model.eval()
    model.to(device)

    output = model(img_var)
    print(output.shape)
    exit()

    stage_indices = ['encoder']
    return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
    model = IntermediateLayerGetter(model, return_layers=return_layers)
    output = model(img_var)
    print([(k, v.shape) for k, v in output.items()])

    return output