import torch
import torchvision
from thop import profile
from geoseg.models.CSAvitNet import CSAvitNet




if __name__ =="__main__":
    print('==> Building model..')
    model = CSAvitNet()
    input=torch.randn(1, 3, 512, 512)
    flops, params = profile(model, (input,))
    print('FLOPs: ', flops, 'params: ', params)
    print('FLOPs: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))