from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import time


# Training Data
def get_data():
    train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
    train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
    dtype = torch.FloatTensor
    X = Variable(torch.from_numpy(train_X).type(dtype),requires_grad=False).view(17,1)
    y = Variable(torch.from_numpy(train_Y).type(dtype),requires_grad=False)
    return X,y

def plot_variable(x,y,z='',**kwargs):
    plt.plot(x.data.numpy(), y.data.numpy(), z, **kwargs)


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 17)


learning_rate = 1e-3
criterion = nn.MSELoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    optimizer.step()
    model.train(True)  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0

    for i in range(10):
        # get the inputs
        inp, labels = get_data()
        inp = inp * i
        labels = labels * i

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward
        i0 = inp[:,0].numpy()
        i1 = np.asarray([i0, i0, i0])
        i2 = np.asarray([i1, i1, i1])
        i3 = np.asarray([i2, i2, i2])
        inputs = torch.from_numpy(i3)
        # breakpoint()
        outputs = model(inputs)
        # _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
            


    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)

inp, labels = get_data()
inp = inp*15
labels = labels*15
i0 = inp[:,0].numpy()
i1 = np.asarray([i0, i0, i0])
i2 = np.asarray([i1, i1, i1])
i3 = np.asarray([i2, i2, i2])
inputs = torch.from_numpy(i3)
# breakpoint()
outputs = model_ft(inputs)
_, preds = torch.max(outputs.data, 1)
loss = criterion(outputs, labels)

print(loss)


def plot_variable(x,y,z='',**kwargs):
    plt.plot(x.data.numpy(), y.data.numpy(), z, **kwargs)

plot_variable(inp,labels,'ro')
plot_variable(inp,outputs[0],'bx')
x,y = get_data()
plot_variable(x,y,'y.')
plt.show()
breakpoint()
