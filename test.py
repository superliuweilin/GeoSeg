import torch
from geoseg.models.MobilevitV2Former import MobilevitV2Former
model = MobilevitV2Former()
model_dict = model.state_dict()
for key in model_dict:
    print(key)
weight_file = "pretrain_weights/mobilevitv2-1.0.pt"  # 权重文件路径
pretrainpth = torch.load(weight_file)
for key in pretrainpth:
    print(key)
