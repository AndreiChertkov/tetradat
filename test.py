# pip install git+https://github.com/RobustBench/robustbench.git


import sys

import torch
import torch.nn as nn

import torchattacks


from torchvision import models
from utils import get_imagenet_data
from robustbench.utils import clean_accuracy


import numpy as np
import matplotlib.pyplot as plt
import json

import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

def get_imagenet_data():
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    # https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    class_idx = json.load(open("./data/imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    imagnet_data = image_folder_custom_label(root='./data/imagenet',
                                             transform=transform,
                                             idx2label=idx2label)
    data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=1, shuffle=False)
    print("Used normalization: mean=", MEAN, "std=", STD)
    return iter(data_loader).next()




images, labels = get_imagenet_data()
print('[Data loaded]')

device = "cpu"
model = models.resnet18(pretrained=True).to(device).eval()
acc = clean_accuracy(model, images.to(device), labels.to(device))
print('[Model loaded]')
print('Acc: %2.2f %%'%(acc*100))
