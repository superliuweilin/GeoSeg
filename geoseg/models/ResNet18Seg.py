import timm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from geoseg.models.ResNet18 import ResNet18

class ConvBNReLU(nn.Sequential):
    def  __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

# backbone_name = 'swsl_resnet18'
# backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
#                              out_indices=(1, 2, 3, 4), pretrained=False)
# print(backbone)
# x = torch.FloatTensor(np.random.rand(2, 3, 512, 512))
# res1, res2, res3, res4 = backbone(x)
# print(res1.shape, res2.shape, res3.shape, res4.shape)

def count_paratermeters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# def load_prepth(pretrain=True):
#         if pretrain:
#             pretrained_dict = torch.load('backbones_pretrain_pth/semi_weakly_supervised_resnet18-118f1556.pth')
#             # print(pretrained_dict)
#             model = ResNet18()
#             model_dict = model.state_dict()
#             for idx, key in enumerate(pretrained_dict.keys()):
#                 for idx_1, key_1 in enumerate(model_dict.keys()):
#                     if key == key_1:
#                         model_dict[key] = pretrained_dict[key]
#                         break
#         model.load_state_dict(model_dict)
#         return model

class ResNet18Seg(nn.Module):
    def __init__(self,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 dropout=0.1,
                 encode_out_channels=512,
                 decode_channels=64,
                 num_classes=6
                 ):
        super().__init__()
        self.backbone = ResNet18()
        self.segmentation_head = nn.Sequential(ConvBNReLU(encode_out_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
    
    


    def forward(self, x):
        h, w = x.size()[-2:]
        _, _, _, res4 = self.backbone(x)
        x = self.segmentation_head(res4)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        return x
    
if __name__ == '__main__':
    model = ResNet18Seg()
    # # print(count_paratermeters(model))
    # # # print(model)
    inputs = torch.FloatTensor(np.random.rand(2, 3, 512, 512))
    out = model(inputs)
    print(out.shape)
    # model = ResNet18()
    # model_dict = model.state_dict
    # # for key in model.state_dict():
    # #     print(key)
    # model_weight = model_dict('conv1.weight')
    # pretrained_dict = torch.load('backbones_pretrain_pth/semi_weakly_supervised_resnet18-118f1556.pth')
    # pre_weight = pretrained_dict['conv1.weight']
    # if torch.allclose(pre_weight, model_weight, atol=1e-4):
    #     print('The pretrained weights are successfully loaded into the model')
    # else:
    #     print('The pretrained weights are not loaded into the model correctly')
    # model = ResNet18()
    # for key in model.state_dict():
    #     print(key)
    # for key in pretrained_dict:
    #     print(key)
    



    
    