import yaml

config_file = '/home/lyu3/lwl_wp/GeoSeg/seg_hrnet_w48_train_ohem_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml'
with open(config_file, 'r') as ifs:
    config = yaml.safe_load(ifs)
print(config)
  # DIST_FC: true
