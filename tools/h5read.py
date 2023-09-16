import h5py
import numpy as np
from PIL import Image

h5_file = "D:/GoProEvent/GOPRO/GOPRO/train/GOPR0372_07_00.h5"
file = h5py.File(h5_file, "r")
groups = [key for key in file.keys()]
print("该文件共有以下几组：", groups)
