import timm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from geoseg.models.MobilevitV2 import mobilevit_s
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from torch.nn import init
from geoseg.models.cshuffleEfficient import EfficientViTBlock


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
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


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class GlobalLocalAttention(nn.Module):
#     def __init__(self,
#                  dim=256,
#                  num_heads=16,
#                  qkv_bias=False,
#                  window_size=8,
#                  relative_pos_embedding=True
#                  ):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // self.num_heads
#         self.scale = head_dim ** -0.5
#         self.ws = window_size

#         self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
#         self.local1 = ConvBN(dim, dim, kernel_size=3)
#         self.local2 = ConvBN(dim, dim, kernel_size=1)
#         self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

#         self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
#         self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

#         self.relative_pos_embedding = relative_pos_embedding

#         if self.relative_pos_embedding:
#             # define a parameter table of relative position bias
#             # 定义一个相对位置偏差的参数表
#             self.relative_position_bias_table = nn.Parameter(
#                 torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

#             # get pair-wise relative position index for each token inside the window
#             coords_h = torch.arange(self.ws)
#             coords_w = torch.arange(self.ws)
#             # torch.arange()返回1维张量
#             coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#             coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#             relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#             relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
#             relative_coords[:, :, 1] += self.ws - 1
#             relative_coords[:, :, 0] *= 2 * self.ws - 1
#             relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#             self.register_buffer("relative_position_index", relative_position_index)

#             trunc_normal_(self.relative_position_bias_table, std=.02)

#     def pad(self, x, ps):
#         _, _, H, W = x.size()
#         if W % ps != 0:
#             x = F.pad(x, (0, ps - W % ps), mode='reflect')
#         if H % ps != 0:
#             x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
#         return x

#     def pad_out(self, x):
#         x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
#         return x

#     def forward(self, x):
#         B, C, H, W = x.shape

#         local = self.local2(x) + self.local1(x)

#         x = self.pad(x, self.ws)
#         B, C, Hp, Wp = x.shape
#         # print(x.shape)
#         qkv = self.qkv(x)
#         # print(qkv.shape)

#         q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
#                             d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

#         dots = (q @ k.transpose(-2, -1)) * self.scale

#         if self.relative_pos_embedding:
#             relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#                 self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
#             relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#             dots += relative_position_bias.unsqueeze(0)

#         attn = dots.softmax(dim=-1)
#         attn = attn @ v

#         attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
#                          d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

#         attn = attn[:, :, :H, :W]

#         out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
#               self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
#         # print(out.shape)
#         # print(local.shape)
#         out = out + local
#         out = self.pad_out(out)
#         out = self.proj(out)
#         # print(out.size())
#         out = out[:, :, :H, :W]
#         # print(out.size())

#         return out
# class GLMobileAttention(nn.Module):
#     def __init__(self,
#                  dim=256,
#                  depth=2,
#                  window_size=8
#                  ):
#         super().__init__()
#         self.local1 = ConvBN(dim, dim, kernel_size=3)
#         self.local2 = ConvBN(dim, dim, kernel_size=1)
#         self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)
#         self.GLB = MobileViTAttention(in_channel=dim, dim=dim, kernel_size=3, patch_size=2, depth=depth, mlp_dim=int(4*dim))
#     def pad_out(self, x):
#         x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
#         return x

#     def forward(self, x):
#         # print(x.shape)
#         local = self.local2(x) + self.local1(x)
#         # print((self.local1(x)).shape)
#         # print((self.local2(x)).shape)
#         out = local + self.GLB(x)
#         out = self.pad_out(out)
#         out = self.proj(out)
#         return out

class MobileViTv2Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model, 1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        '''
        i = self.fc_i(input)  # (bs,nq,1)
        weight_i = torch.softmax(i, dim=1)  # bs,nq,1
        context_score = weight_i * self.fc_k(input)  # bs,nq,d_model
        context_vector = torch.sum(context_score, dim=1, keepdim=True)  # bs,1,d_model
        v = self.fc_v(input) * context_vector  # bs,nq,d_model
        out = self.fc_o(v)  # bs,nq,d_model

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=4, window_size=7, resolution=64):
        super().__init__()

        # self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)
        self.attn = EfficientViTBlock('s',
                                      dim, 16, num_heads,
                                      resolution,
                                      window_size,
                                      [5, 5, 5, 5])
        # self.local1 = ConvBN(dim, dim, kernel_size=3)
        # self.local2 = ConvBN(dim, dim, kernel_size=1)

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # # DropPath 若x为输入的张量，其通道为[B,C,H,W]，那么drop_path的含义为在一个Batch_size中，随机有drop_prob的样本，
        # # 不经过主干，而直接由分支进行恒等映射。
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
        #                drop=drop)
        # self.norm2 = norm_layer(dim)

    def forward(self, x):
        # print((self.norm1(x)).shape)
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print(x.shape)
        # local = self.local1(x) + self.local2(x)
        x = self.attn(x)
        # print(x.shape)
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # torch.nn.Parameter可以理解为类型转换函数，将一个不可训练的Tensor转换为可训练的参数
        # torch.ones()返回一个全为1的张量，形状由参数size定义。
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


# class AddFuse(nn.Module):
#     def __init__(self, in_channels=128, decode_channels=128):
#         super(AddFuse, self).__init__()
#         self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
#         # self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
#
#     def forward(self, x, res):
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
#         # weights = nn.ReLU()(self.weights)
#         # fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#         # x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
#         x = x + self.pre_conv(res)
#         # x = self.post_conv(x)
#         return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels // 16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels // 16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


# class AuxHead(nn.Module):

#     def __init__(self, in_channels=64, num_classes=8):
#         super().__init__()
#         self.conv = ConvBNReLU(in_channels, in_channels)
#         self.drop = nn.Dropout(0.1)
#         self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

#     def forward(self, x, h, w):
#         feat = self.conv(x)
#         feat = self.drop(feat)
#         feat = self.conv_out(feat)
#         feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
#         return feat
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        inner_dim = heads * head_dim
        project_out = not (heads == 1 and head_dim == dim)
        '''如果head==1并且head_dim==dim,project_out=False,否则project_out=True'''

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # print(x.shape)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # print('qkv', qkv[0].shape)
        # chunk对数组进行分组,3表示组数，dim表示在哪一个维度.这里返回的qkv为一个包含三组张量的元组
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        # print('self.to_out', self.to_out(out).shape)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim,Attention(dim, heads=8, head_dim=64, dropout=dropout)),
                PreNorm(dim, MobileViTv2Attention(dim)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        out = x
        for att, ffn in self.layers:
            out = out + att(out)
            out = out + ffn(out)
        return out


class MobileViTAttention(nn.Module):
    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=2, depth=3, mlp_dim=1024):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)

        self.trans = Transformer(dim=dim, depth=depth, heads=8, head_dim=64, mlp_dim=mlp_dim)

        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)
        self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        y = x.clone()  # bs,c,h,w
        # y = x

        ## Local Representation
        y = self.conv2(self.conv1(x))  # bs,dim,h,w

        ## Global Representation
        _, _, h, w = y.shape
        # print(y.shape)
        y = rearrange(y, 'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim', ph=self.ph, pw=self.pw)  # bs,h,w,dim
        # print('y_a',y.shape)
        y = self.trans(y)
        # print('y_at', y.shape)
        y = rearrange(y, 'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)', ph=self.ph, pw=self.pw, nh=h // self.ph,
                      nw=w // self.pw)  # bs,dim,h,w

        ## Fusion
        y = self.conv3(y)  # bs,dim,h,w
        y = torch.cat([x, y], 1)  # bs,2*dim,h,w
        y = self.conv4(y)  # bs,c,h,w

        return y


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=7,
                 num_classes=16,
                 resolutions=[128, 64, 32, 16]
                 ):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=4, window_size=window_size, resolution=resolutions[-1])

        self.b3 = Block(dim=decode_channels, num_heads=4, window_size=window_size, resolution=resolutions[-2])
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=4, window_size=window_size, resolution=resolutions[-3])
        self.p2 = WF(encoder_channels[-3], decode_channels)

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            # self.aux_head = AuxHead(decode_channels, num_classes)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        if self.training:
            # print('Res4 = ', (self.pre_conv(res4)).shape)
            x = self.b4(self.pre_conv(res4))
            # print('x = ', x.shape)
            # h4 = self.up4(x)

            x = self.p3(x, res3)
            # print(x.shape)
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # x = x + res3
            x = self.b3(x)
            # print(x.shape)
            # h3 = self.up3(x)

            x = self.p2(x, res2)
            # print(x.shape)
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # x = x + res2
            x = self.b2(x)
            # print(x.shape)
            # h2 = x
            x = self.p1(x, res1)
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # x = x + res1
            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            # ah = h4 + h3 + h2
            # ah = self.aux_head(ah, h, w)

            return x
        else:
            x = self.b4(self.pre_conv(res4))
            x = self.p3(x, res3)
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # x = x + res3
            x = self.b3(x)
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            x = self.p2(x, res2)
            # x = x + res2
            x = self.b2(x)
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # x = x + res1

            x = self.p1(x, res1)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CSAvitNet(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 pretrained=True,
                 window_size=7,
                 num_classes=6,
                 resolutions=[128, 64, 32, 16]
                 ):
        super().__init__()
        self.backbone = mobilevit_s()

        encoder_channels = (64, 96, 128, 640)
        # print(encoder_channels)

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes,
                               resolutions=resolutions)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        # print(res1.shape, res2.shape, res3.shape, res4.shape)
        if self.training:
            # x, ah = self.decoder(res1, res2, res3, res4, h, w)
            # return x, ah
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x


if __name__ == '__main__':
    model = CSAvitNet()
    model = model.eval()
    # # print(count_paratermeters(model))
    # # # print(model)
    inputs = torch.FloatTensor(np.random.rand(2, 3, 896, 896))
    out = model(inputs)
    print(out.shape)
