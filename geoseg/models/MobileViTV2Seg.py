import timm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from geoseg.models.Mobilevit import mobilevit_s

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

class MobilevitV2Seg(nn.Module):
    def __init__(self,
                 pretrained=True,
                 dropout=0.1,
                 encode_out_channels=640,
                 decode_channels=64,
                 num_classes=6
                 ):
        super().__init__()
        self.backbone = mobilevit_s()
        self.segmentation_head = nn.Sequential(ConvBNReLU(encode_out_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
    
    


    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        res1 = F.interpolate(res1, scale_factor=2, mode='bilinear', align_corners=False)
        res2 = F.interpolate(res2, scale_factor=2, mode='bilinear', align_corners=False)
        res3 = F.interpolate(res3, scale_factor=2, mode='bilinear', align_corners=False)
        x = res1 + res2 + res3 + res4
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        return x
    
if __name__ == '__main__':
    model = MobilevitV2Seg()
    # # print(count_paratermeters(model))
    # # # print(model)
    inputs = torch.FloatTensor(np.random.rand(2, 3, 512, 512))
    out = model(inputs)
    print(out.shape)