from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.potsdam_dataset import *
from geoseg.models.PVTSeg import PVTSeg
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 300
ignore_index = len(CLASSES)
train_batch_size = 16
val_batch_size = 16
lr = 0.01
weight_decay = 0.0005
backbone_lr = 0.01
backbone_weight_decay = 0.0005
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

test_time_aug = 'd4'
output_mask_dir, output_mask_rgb_dir = None, None
weights_name = "PVTSeg_512_300epoch"
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = "PVTSeg_512_300epoch"
log_name = 'potsdam/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
gpus = [1]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = 'model_weights/potsdam/PVTSeg_512_300epoch/last.ckpt'
#  define the network
net = PVTSeg(num_classes=num_classes)

# define the loss
loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
use_aux_loss = False

# define the dataloader

train_dataset = PotsdamDataset(data_root='/home/lyu/datasets/Potsdam/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset = PotsdamDataset(transform=val_aug)
test_dataset = PotsdamDataset(data_root='/home/lyu/datasets/Potsdam/test',
                              transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.SGD(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-5)

