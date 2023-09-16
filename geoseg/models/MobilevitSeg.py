from geoseg.models.Mobilevit import mobilevit_s
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    def  __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class Decoder(nn.Module):
    def __init__(self, 
                 encoder_channels=640,
                 decode_channels=64, 
                 dropout=0.1,
                 num_classes=6):
        super(Decoder, self).__init__()
        self.segmentation_head = nn.Sequential(ConvBNReLU(encoder_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()
    def forward(self, x, h, w):
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x
    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
class MobilevitSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mobilevit_s()
        self.decoder = Decoder()
    def forward(self, x):
        # print(x.shape)
        _, _, h, w = x.size()
        _, _, _, x = self.backbone(x)
        x = self.decoder(x, h, w)
        return x
if __name__ == '__main__':
    model = MobilevitSeg()
    print(model)

    inputs = torch.FloatTensor(np.random.rand(1, 3, 512, 512))
    out = model(inputs)
    print(out.shape)