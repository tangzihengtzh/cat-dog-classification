import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2

# print(os.getcwd())
root_dir=r"D:\pythonItem\catdog\catsdogs\train"
for idx, category in enumerate(['cat', 'dog']):
    category_path = os.path.join(root_dir, category)
    for img_name in os.listdir(category_path):
        img_names = []
        if category == "dog":
            labels = torch.tensor([1,0])
        else:
            labels = torch.tensor([0,1])
        img_names.append(os.path.join(category_path, img_name))
        # labels.append(idx)
        print(idx, category)
        print(img_names)
        print(labels)