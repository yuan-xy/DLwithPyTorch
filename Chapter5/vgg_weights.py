import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pickle, time
from torchvision.datasets import ImageFolder
import numpy as np
from torchvision import models

def plot_img(image):
    image = image.numpy()[0]
    mean = 0.1307
    std = 0.3081
    image = ((mean * image) + std)
    plt.imshow(image,cmap='gray')

def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


vgg = models.vgg16(pretrained=True)
for key in vgg.state_dict().keys():
    print(key)
    cnn_weights = vgg.state_dict()[key].cpu()
    print(cnn_weights.shape)
    fig = plt.figure()
    fig.subplots_adjust(left=0,right=1,bottom=0,top=0.9,hspace=0.1,wspace=0.2)
    for i in range(64):
        ax = fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
        imshow(cnn_weights[i])
    break

plt.show()
breakpoint()



