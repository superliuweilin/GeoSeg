import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from geoseg.models.ECA_module import eca_layer
class DepthWiseConv(nn.Module):
    def __init__(self,in_channel,out_channel, kernel_size=3, stride=1, padding=0, bias=False):
 
        #这一行千万不要忘记
        super(DepthWiseConv, self).__init__()
 
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_channel,
                                    bias=bias)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积
 
        #逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1], padding=1, k_size=3) -> None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.avt1 = nn.ReLU(inplace=True)  # 原地替换 节省内存开销
        self.conv2 = DepthWiseConv(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.avt2 = nn.ReLU(inplace=True)
        self.eca = eca_layer(out_channels, k_size=k_size)

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.downsample = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.avt2(out)
        out = self.eca(out)
        out = out + self.downsample(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, k_size=[3, 5, 5, 5]):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, 6)

        # conv2_x
        self.layer1 = self._make_layer(BasicBlock, 64, [[1, 1], [1, 1]], k_size=int(k_size[0]))
        # self.conv2_2 = self._make_layer(BasicBlock,64,[1,1])

        # conv3_x
        self.layer2 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]], k_size=int(k_size[1]))
        # self.conv3_2 = self._make_layer(BasicBlock,128,[1,1])

        # conv4_x
        self.layer3 = self._make_layer(BasicBlock, 256, [[2, 1], [1, 1]], k_size=int(k_size[2]))
        # self.conv4_2 = self._make_layer(BasicBlock,256,[1,1])

        # conv5_x
        self.layer4 = self._make_layer(BasicBlock, 512, [[2, 1], [1, 1]], k_size=int(k_size[3]))
        # self.conv5_2 = self._make_layer(BasicBlock,512,[1,1])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, 1000)

    # 这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides, k_size):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, k_size=k_size))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out1 = out
        out = self.layer2(out)
        out2 = out
        out = self.layer3(out)
        out3 = out
        out = self.layer4(out)
        out4 = out

        # out = self.avgpool(out)
        # out = out.reshape(x.shape[0], -1)
        # out = self.fc(out)

        return out1, out2, out3, out4
        # return out
def count_paratermeters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':
    model = ResNet18()
    print(count_paratermeters(model))
    # print(model)
    inputs = torch.FloatTensor(np.random.rand(2, 3, 512, 512))
    out = model(inputs)
    for idex in range(len(out)):
        print(out[idex].shape)

