import torch
import torch.nn as nn
class mtcaconv2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        feat = dim // 8
        self.ldim = 2*feat
        self.bdim = 2* feat
        self.sdim = 3* feat
        self.cdim =  feat
        self.lconv = nn.Conv2d(2*feat, dim, 7, 1, 3,groups=2*feat)
        self.bconv = nn.Conv2d(2*feat, dim, 5, 1, 2,groups=2*feat)
        self.sconv = nn.Conv2d(3*feat, dim, 3, 1, 1,groups=feat)
        self.lp = nn.Conv2d(dim,dim,1,1,0)
        self.bp = nn.Conv2d(dim, dim, 1, 1, 0)
        self.sp = nn.Conv2d(dim, dim, 1, 1, 0)
        self.upl = nn.Conv2d(feat, dim, 1, 1, 0)

        self.w = nn.Parameter(torch.ones(3))

    def forward(self,x):
        l, b, s, c = torch.split(x,(self.ldim,self.bdim,self.sdim,self.cdim),dim=1)
        # print(self.m,self.n,self.p)
        l = self.lp(self.lconv(l))
        b = self.bp(self.bconv(b))
        s = self.sp(self.sconv(s))
        cl = self.upl(c)
        l = cl*l
        b = cl*b
        s = cl*s
        # 归一化权重
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))

        cl = l*w1+b*w2+s*w3
        # x = self.OUT(cl)
        # print(self.m[0], self.n[0], self.p[0])
        return cl