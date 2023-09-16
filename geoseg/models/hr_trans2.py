import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.xyconv import CAM

BN_MOMENTUM = 0.1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=False, k_size=3):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        ds = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        )
        self.ds = downsample
        self.downsample = ds
        self.stride = stride

    def forward(self, x):
        # print('x = {}'.format(x.shape))
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.eca(out)
        # print('out = {}'.format(out.shape))

        if self.ds:
            residual = self.downsample(x)
            # print('residual = {}'.format(residual.shape))

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c, stage=2, rep=1):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()

        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            stride = 1
            stages = 1
            ds = False
            if stage > 2 and i == 1:
                stages = 2
                stride = 2
                ds = True
                if rep == 1:
                    branch = nn.Sequential(
                        BasicBlock(w * (stage - 2), w * (stages ** (stage - 2)), stride=stride, downsample=ds),
                        BasicBlock(w * (stages ** (stage - 2)), w * (stages ** (stage - 2))),
                        BasicBlock(w * (stages ** (stage - 2)), w * (stages ** (stage - 2))),
                        BasicBlock(w * (stages ** (stage - 2)), w * (stages ** (stage - 2)))
                    )
                else:
                    ds = False
                    stride = 1
                    branch = nn.Sequential(
                        BasicBlock(w * (stages ** (stage - 2)), w * (stages ** (stage - 2)), stride=stride,
                                   downsample=ds),
                        BasicBlock(w * (stages ** (stage - 2)), w * (stages ** (stage - 2))),
                        BasicBlock(w * (stages ** (stage - 2)), w * (stages ** (stage - 2))),
                        BasicBlock(w * (stages ** (stage - 2)), w * (stages ** (stage - 2)))
                    )

            else:
                branch = nn.Sequential(
                    BasicBlock(w * stages, w * stages, stride=stride, downsample=ds),
                    BasicBlock(w * stages, w * stages),
                    BasicBlock(w * stages, w * stages),
                    BasicBlock(w * stages, w * stages)
                )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        stages = 1
        if stage == 3:
            stages = 2
        elif stage == 4:
            stages = 4
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j) * stages, c * (2 ** i), kernel_size=1, stride=1,
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i) * stages, mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    # for k in range(i - j - 1):
                    #     ops.append(
                    #         nn.Sequential(
                    #             nn.Conv2d(c * (2 ** j) * stages, c * (2 ** j) * stages, kernel_size=3, stride=2, padding=1, bias=False),
                    #             nn.BatchNorm2d(c * (2 ** j) * stages, momentum=BN_MOMENTUM),
                    #             nn.ReLU(inplace=True)
                    #         )
                    #     )
                    # # 最后一个卷积层不仅要调整通道，还要进行下采样
                    if stage == 2:
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j) * stages, c * (2 ** i) * stages, kernel_size=3, stride=2,
                                          padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** i) * stages, momentum=BN_MOMENTUM)
                            )
                        )
                    elif stage == 3:
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** i) * stages, kernel_size=3, stride=2,
                                          padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** i) * stages, momentum=BN_MOMENTUM),
                                nn.Conv2d(c * (2 ** i) * stages, c * (2 ** i) * stages, kernel_size=3, stride=2,
                                          padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** i) * stages, momentum=BN_MOMENTUM)
                            )
                        )
                    elif stage == 4:
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** i) * stages, kernel_size=3, stride=2,
                                          padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** i) * stages, momentum=BN_MOMENTUM),
                                nn.Conv2d(c * (2 ** i) * stages, c * (2 ** i) * stages, kernel_size=3, stride=2,
                                          padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** i) * stages, momentum=BN_MOMENTUM),
                                nn.Conv2d(c * (2 ** i) * stages, c * (2 ** i) * stages, kernel_size=3, stride=2,
                                          padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** i) * stages, momentum=BN_MOMENTUM)
                            )
                        )

                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        # print(len(self.fuse_layers))

        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )

        return x_fused


class HighResolutionNet(nn.Module):
    def __init__(self, base_channel: int = 32, class_num: int = 6):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.cam1 = CAM(288, 288, 32)

        # Stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(  # 这里又使用一次Sequential是为了适配原项目中提供的权重
                    nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel, stage=2)
        )

        # transition2
        # self.transition2 = nn.ModuleList([
        #     nn.Identity(),  # None,  - Used in place of "None" because it is callable
        #     nn.Identity(),  # None,  - Used in place of "None" because it is callable
        #     nn.Sequential(
        #         nn.Sequential(
        #             nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
        #             nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
        #             nn.ReLU(inplace=True)
        #         )
        #     )
        # ])

        # Stage3
        self.stage3 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel, stage=3, rep=1),
            StageModule(input_branches=2, output_branches=2, c=base_channel, stage=3, rep=2),
            StageModule(input_branches=2, output_branches=2, c=base_channel, stage=3, rep=3),
            StageModule(input_branches=2, output_branches=2, c=base_channel, stage=3, rep=4)
        )

        # transition3
        # self.transition3 = nn.ModuleList([
        #     nn.Identity(),  # None,  - Used in place of "None" because it is callable
        #     nn.Identity(),  # None,  - Used in place of "None" because it is callable
        #     nn.Identity(),  # None,  - Used in place of "None" because it is callable
        #     nn.Sequential(
        #         nn.Sequential(
        #             nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
        #             nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
        #             nn.ReLU(inplace=True)
        #         )
        #     )
        # ])

        # Stage4

        self.stage4 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel, stage=4, rep=1),
            StageModule(input_branches=2, output_branches=2, c=base_channel, stage=4, rep=2),
            StageModule(input_branches=2, output_branches=2, c=base_channel, stage=4, rep=3)
        )

        # Final layer
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=class_num,
                kernel_size=1,
                stride=1,
                padding=0)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)

        x = [trans(x) for trans in self.transition1]  # Since now, x is a list

        x = self.stage2(x)

        #     self.transition2[0](x[0]),
        #     self.transition2[1](x[1]),
        #     self.transition2[2](x[-1])
        # ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)

        # x0 = torch.Tensor(x[0])
        # x1 = torch.Tensor(x[1])
        # print(x0.shape)
        # print(x1.shape)
        # x = [
        #     self.transition3[0](x[0]),
        #     self.transition3[1](x[1]),
        #     self.transition3[2](x[2]),
        #     self.transition3[3](x[-1]),
        # ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        # upsample
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        # print(x0_w, x0_h)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x = torch.cat([x[0], x1], 1)

        # print(x.shape)
        x = self.final_layer(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x


if __name__ == '__main__':
    print(HighResolutionNet())
    model = HighResolutionNet()
    inputs = torch.FloatTensor(np.random.rand(1, 3, 512, 512))
    out = model(inputs)
    print(out.shape)

    # print(out[0].shape)
    # print(out[1].shape)
