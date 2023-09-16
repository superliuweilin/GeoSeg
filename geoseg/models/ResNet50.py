import timm
import torch
import numpy as np
import torch.nn as nn

from geoseg.models.UNetFormer import ConvBNReLU


class ResNet50(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet50, self).__init__()
        backbone_name = 'resnet50'
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=[4], pretrained=False)
        self.conv2 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.segmentation_head = nn.Sequential(ConvBNReLU(512, 512),
                                               nn.Dropout2d(p=0.1, inplace=True),
                                               nn.Conv2d(512, num_classes, kernel_size=1))

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv2(x[0])
        x = self.up1(x)
        x = self.segmentation_head(x)
        x = self.up2(x)
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x


if __name__ == '__main__':
    x = torch.FloatTensor(np.random.rand(1, 3, 1024, 1024))
    model = ResNet50()
    print(model)
    out = model(x)
    print(out.shape)
