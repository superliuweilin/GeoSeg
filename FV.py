import contextlib
import math
from pathlib import Path
import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version
from scipy.ndimage import gaussian_filter1d
from torchvision import transforms
from geoseg.models.PVTSeg import PVTSeg

# from ultralytics.yolo.utils import LOGGER, TryExcept, plt_settings, threaded

# from .checks import check_font, check_version, is_ascii
# from .files import increment_path
# from .ops import clip_boxes, scale_image, xywh2xyxy, xyxy2xywh

def feature_visualization(x, module_type, stage, n=32, save_dir=Path('/home/lyu/lwl_wsp/GeoSeg/visualization')):
    """
    Visualize feature maps of a given model module during inference.

    Args:
        x (torch.Tensor): Features to be visualized.
        module_type (str): Module type.
        stage (int): Module stage within the model.
        n (int, optional): Maximum number of feature maps to plot. Defaults to 32.
        save_dir (Path, optional): Directory to save results. Defaults to Path('runs/detect/exp').
    """
    # for m in ['Detect', 'Pose', 'Segment']:
    #     if m in module_type:
    #         return
    batch, channels, height, width = x.shape  # batch, channels, height, width
    if height > 1 and width > 1:
        f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_LVTSeg.png"  # filename
        # f = '/home/lyu3/lwl_wp/GeoSeg/visualization/test.png'

        blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
        n = min(n, channels)  # number of plots
        fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(n):
            ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
            ax[i].axis('off')

        # LOGGER.info(f'Saving {f}... ({n}/{channels})')
        plt.savefig(f, dpi=300, bbox_inches='tight')
        plt.close()
        np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # npy save

image_path = '/home/lyu/datasets/Potsdam/test/images_1024/top_potsdam_2_13_0_2.tif'
image = Image.open(image_path).convert('RGB')
# dpi = image.info.get('dpi', 100)
# print(dpi)
# preprocess = transforms.Compose([transforms.RandomCrop(512), transforms.ToTensor()])
preprocess = transforms.Compose([transforms.ToTensor()])
input_tensor = preprocess(image)
# print(input_tensor)
input_batch = input_tensor.unsqueeze(0)

model = PVTSeg()
# 加载检查点文件
checkpoint = torch.load('/home/lyu/lwl_wsp/GeoSeg/model_weights/potsdam/PVTSeg_512_300epoch/PVTSeg_512_300epoch.ckpt')
# for key in checkpoint['state_dict']:
#     print(key)
# # # 去除权重键名的前缀
# state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items()}

has_net_prefix = any(k.startswith('net.') for k in checkpoint['state_dict'].keys())
if has_net_prefix:
    state_dict = {k.replace('net.', '', 1): v for k, v in checkpoint['state_dict'].items()}
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('backbone.pos_embed')}

# state_dict = checkpoint['state_dict']
# for Key in state_dict:
#     print(Key)
buffer = io.BytesIO()
torch.save(state_dict, buffer)

# 重新定位缓冲区的指针位置
buffer.seek(0)

# model = net.load_from_checkpoint(buffer)
# # 替换'path_to_weight_file'为你的权重文件路径
# model.load_state_dict(torch.load('/home/lyu3/lwl_wp/GeoSeg/model_weights/potsdam/CSAvitNet_512_300epoch_FRH/CSAvitNet_512_300epoch_FRH.ckpt'))
state_dict = torch.load(buffer)
model.load_state_dict(state_dict)
model.eval()
print(input_batch.shape)
with torch.no_grad():
    x = model(input_batch)
feature_visualization(x, 'Segment', stage=32)